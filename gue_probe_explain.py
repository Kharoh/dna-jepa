#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from datasets import load_dataset
from transformers import AutoTokenizer


# -------------------------
# 1) Backbone (same as yours)
# -------------------------
class ALiBi(nn.Module):
    def __init__(self, num_heads, max_seq_len=4096):
        super().__init__()
        slopes = torch.tensor([2 ** (-8 / num_heads * (i + 1)) for i in range(num_heads)])
        positions = torch.arange(max_seq_len).unsqueeze(0) - torch.arange(max_seq_len).unsqueeze(1)
        alibi = slopes.unsqueeze(-1).unsqueeze(-1) * positions.abs()
        self.register_buffer("alibi", -alibi)

    def forward(self, seq_len):
        return self.alibi[:, :seq_len, :seq_len]


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(GEGLU(dim, hidden_dim), nn.Linear(hidden_dim, dim))

    def forward(self, x, alibi_bias, padding_mask=None):
        B, L, D = x.shape
        residual = x

        x = self.norm1(x)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = alibi_bias.unsqueeze(0)  # (1, H, L, L)
        if padding_mask is not None:
            # padding_mask is boolean broadcastable to (B, 1, 1, L)
            float_mask = torch.zeros_like(padding_mask, dtype=x.dtype).masked_fill(~padding_mask, float("-inf"))
            attn_mask = attn_mask + float_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        return x + residual + self.mlp(self.norm2(out + residual))


class DNAEncoder(nn.Module):
    def __init__(self, tokenizer, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.embed = nn.Embedding(tokenizer.vocab_size, dim, padding_idx=self.pad_id)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.alibi = ALiBi(num_heads)
        self.blocks = nn.ModuleList([Block(dim, num_heads=num_heads) for _ in range(depth)])

        # Not used for probing (we probe CLS features), but kept for ckpt compatibility
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, return_feats=False):
        B, L = x.shape
        padding_mask = (x != self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)

        x = self.embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_mask = torch.ones((B, 1, 1, 1), device=x.device, dtype=torch.bool)
        padding_mask = torch.cat((cls_mask, padding_mask), dim=-1)  # (B,1,1,L+1)

        alibi_bias = self.alibi(L + 1)

        for block in self.blocks:
            x = block(x, alibi_bias, padding_mask)

        cls_out = x[:, 0]
        if return_feats:
            return cls_out
        return self.proj(cls_out)


# -------------------------
# 2) Probe + utilities
# -------------------------
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(self.bn(x))


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_gue(batch, tokenizer, max_len):
    seqs = [b["sequence"] for b in batch]
    labels = [int(b["label"]) for b in batch]

    toks_list = []
    for s in seqs:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        toks_list.append(torch.tensor(ids, dtype=torch.long))

    return torch.stack(toks_list, dim=0), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def extract_features(backbone, loader, device, amp_dtype=torch.bfloat16):
    backbone.eval()
    feats, ys = [], []
    for toks, y in loader:
        toks = toks.to(device, non_blocking=True)
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=(device.startswith("cuda"))):
            f = backbone(toks, return_feats=True)
        feats.append(f.float().cpu())
        ys.append(y.cpu())
    return torch.cat(feats, 0), torch.cat(ys, 0)


def train_probe(train_feats, train_y, num_classes, device, epochs=10, lr=1e-3, bs=256, wd=1e-2):
    probe = LinearProbe(train_feats.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(train_feats, train_y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for fx, fy in dl:
            fx, fy = fx.to(device), fy.to(device)
            loss = crit(probe(fx), fy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def eval_probe(probe, test_feats, test_y, device):
    probe.eval()
    logits = []
    bs = 1024
    for i in range(0, len(test_feats), bs):
        fx = test_feats[i:i+bs].to(device)
        logits.append(probe(fx).float().cpu())
    logits = torch.cat(logits, 0)
    preds = logits.argmax(1).numpy()
    y = test_y.numpy()
    acc = (preds == y).mean()
    return acc, logits


# -------------------------
# 3) t-SNE plot
# -------------------------
def plot_tsne(feats, labels, out_png, seed=0):
    X = feats.numpy()
    y = labels.numpy()

    # Practical speed-up: PCA -> tSNE
    pca = PCA(n_components=min(50, X.shape[1]), random_state=seed)
    Xp = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=seed)
    Z = tsne.fit_transform(Xp)

    plt.figure(figsize=(7, 6), dpi=160)
    for cls in sorted(np.unique(y)):
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.75, label=f"class {cls}")
    plt.title("t-SNE of backbone CLS embeddings (test set)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# 4) Random-drop importance (uncertainty gap)
# -------------------------
def entropy_from_probs(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def predict_proba(backbone, probe, toks_batch, device, amp_dtype=torch.bfloat16, batch_size=64):
    """
    Predicts probabilities in mini-batches to avoid OOM when mc_samples is large.
    """
    backbone.eval()
    probe.eval()
    
    all_probs = []
    
    # Process inputs in smaller chunks (e.g., 64 at a time)
    for i in range(0, len(toks_batch), batch_size):
        chunk = toks_batch[i : i + batch_size].to(device)
        
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=(device.startswith("cuda"))):
            feats = backbone(chunk, return_feats=True)
            logits = probe(feats)
            probs = torch.softmax(logits.float(), dim=-1)
        
        all_probs.append(probs.cpu())
    
    return torch.cat(all_probs, dim=0).numpy()



def token_spans_to_nucleotides(tokenizer, token_ids):
    """
    Builds:
      - pieces: decoded piece for each token id (string of A/C/G/T/N possibly empty)
      - spans: (start,end) indices in the reconstructed nucleotide string
      - nuc_seq: reconstructed nucleotide string
    """
    pieces = []
    for tid in token_ids:
        if tid == tokenizer.pad_token_id:
            pieces.append("")
            continue
        # decode single token; keep only A/C/G/T/N
        s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False, skip_special_tokens=True)
        s = re.sub(r"[^ACGTNacgtn]", "", s).upper()
        pieces.append(s)

    spans = []
    pos = 0
    for s in pieces:
        start = pos
        pos += len(s)
        end = pos
        spans.append((start, end))

    nuc_seq = "".join(pieces)
    return pieces, spans, nuc_seq


def importance_random_drop(
    backbone, probe, tokenizer, seq_str, label,
    device, max_len=512, mc_samples=256, drop_prob=0.2, seed=0,
):
    set_seed(seed)

    ids = tokenizer(seq_str, add_special_tokens=False)["input_ids"][:max_len]
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    ids = torch.tensor(ids, dtype=torch.long)

    # baseline
    p0 = predict_proba(backbone, probe, ids.unsqueeze(0), device=device)[0]
    u0 = entropy_from_probs(p0)

    L = ids.shape[0]
    valid = (ids != tokenizer.pad_token_id).numpy().astype(bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError("Sequence tokenized to all PAD.")

    # Build MC batch of perturbed inputs
    toks = ids.unsqueeze(0).repeat(mc_samples, 1)

    # sample masks: True = drop
    drop = (torch.rand((mc_samples, L)) < drop_prob)
    # never drop PAD positions
    drop[:, torch.tensor(~valid, dtype=torch.bool)] = False
    # ensure at least 1 token kept per sample
    kept_counts = (~drop[:, valid_idx]).sum(dim=1)
    bad = (kept_counts == 0).nonzero(as_tuple=False).squeeze(-1)
    if bad.numel() > 0:
        # force keep one random valid token
        for b in bad.tolist():
            j = int(valid_idx[np.random.randint(0, len(valid_idx))])
            drop[b, j] = False

    toks[drop] = tokenizer.pad_token_id

    probs = predict_proba(backbone, probe, toks, device=device)  # (M, C)
    ent = np.array([entropy_from_probs(p) for p in probs], dtype=np.float32)

    # Conditional entropy gap per token
    I = np.zeros((L,), dtype=np.float32)
    for i in range(L):
        if not valid[i]:
            I[i] = 0.0
            continue
        di = drop[:, i].cpu().numpy().astype(bool)
        if di.sum() < 5 or (~di).sum() < 5:
            I[i] = 0.0
        else:
            I[i] = float(ent[di].mean() - ent[~di].mean())

    # Map token importance -> nucleotide importance by distributing score over decoded piece length
    pieces, spans, nuc_seq = token_spans_to_nucleotides(tokenizer, ids.tolist())
    nuc_imp = np.zeros((len(nuc_seq),), dtype=np.float32)

    for i, (s, (a, b)) in enumerate(zip(pieces, spans)):
        if b <= a or len(s) == 0:
            continue
        nuc_imp[a:b] += I[i] / max(1, (b - a))

    return {
        "token_ids": ids.tolist(),
        "baseline_probs": p0.tolist(),
        "baseline_entropy": float(u0),
        "token_importance": I.tolist(),
        "nuc_seq": nuc_seq,
        "nuc_importance": nuc_imp.tolist(),
        "label": int(label),
    }


def plot_sequence_importance(nuc_seq, nuc_importance, out_png, title, max_show=None):
    if max_show is not None:
        nuc_seq = nuc_seq[:max_show]
        imp = np.array(nuc_importance[:max_show], dtype=np.float32)
    else:
        imp = np.array(nuc_importance, dtype=np.float32)

    if len(nuc_seq) == 0:
        return

    vmax = float(np.quantile(np.abs(imp), 0.98) + 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    if max_show is not None:
        fig = plt.figure(figsize=(max(10, max_show / 18), 2.8), dpi=180)
    else:
        seq_len = len(nuc_seq)
        fig = plt.figure(figsize=(max(10, seq_len / 18), 2.8), dpi=180)
    ax = plt.gca()

    # 1xN heatmap background
    ax.imshow(imp[None, :], aspect="auto", cmap="coolwarm", norm=norm, extent=[0, len(nuc_seq), 0, 1])

    # nucleotide letters
    for i, ch in enumerate(nuc_seq):
        ax.text(i + 0.5, 0.5, ch, ha="center", va="center", fontsize=8, family="monospace", color="black")

    ax.set_xlim(0, len(nuc_seq))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, len(nuc_seq), num=9))
    ax.set_xlabel("Nucleotide position (truncated view)")
    ax.set_title(title)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Importance (Δ conditional entropy)")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# -------------------------
# 5) Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoints/step_XXXX.pth")
    ap.add_argument("--task", type=str, default="human_tf_0", help="GUE config name (e.g., human_tf_0)")
    ap.add_argument("--outdir", type=str, default="gue_probe_outputs")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_len", type=int, default=512)

    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=128)

    ap.add_argument("--probe_epochs", type=int, default=10)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_bs", type=int, default=256)

    ap.add_argument("--tsne", action="store_true", help="Make t-SNE plot for test embeddings")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--explain_k", type=int, default=6, help="How many test sequences to explain")
    ap.add_argument("--mc_samples", type=int, default=256)
    ap.add_argument("--drop_prob", type=float, default=0.2)
    ap.add_argument("--explain_max_show", type=int, default=None)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # Tokenizer (same as your training)
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir="./"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Backbone
    backbone = DNAEncoder(
        tokenizer, dim=args.dim, depth=args.depth, num_heads=args.heads, proj_dim=args.proj_dim
    ).to(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    # FIX: Clean the state_dict keys to remove torch.compile prefix
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "_orig_mod." prefix if present
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
        
    backbone.load_state_dict(new_state_dict, strict=True)
    backbone.eval()


    # Dataset (GUE)
    ds = load_dataset("leannmlindsey/GUE", args.task, cache_dir="./")
    num_classes = len(set(int(x["label"]) for x in ds["train"]))

    train_dl = DataLoader(
        ds["train"], batch_size=64, shuffle=False,
        collate_fn=lambda b: collate_gue(b, tokenizer, args.max_len),
        num_workers=2, pin_memory=True,
    )
    test_dl = DataLoader(
        ds["test"], batch_size=64, shuffle=False,
        collate_fn=lambda b: collate_gue(b, tokenizer, args.max_len),
        num_workers=2, pin_memory=True,
    )

    # Extract features once
    train_feats, train_y = extract_features(backbone, train_dl, device=args.device)
    test_feats, test_y = extract_features(backbone, test_dl, device=args.device)

    # Train probe
    probe = train_probe(
        train_feats.to(args.device), train_y.to(args.device),
        num_classes=num_classes, device=args.device,
        epochs=args.probe_epochs, lr=args.probe_lr, bs=args.probe_bs,
    )
    acc, _ = eval_probe(probe, test_feats, test_y, device=args.device)
    print(f"[{args.task}] Test accuracy (linear probe): {acc:.4f}")

    # Save probe
    probe_path = os.path.join(args.outdir, f"probe_{args.task}.pth")
    torch.save({"probe_state_dict": probe.state_dict(), "num_classes": num_classes}, probe_path)

    # t-SNE
    if args.tsne:
        tsne_png = os.path.join(args.outdir, f"tsne_{args.task}.png")
        plot_tsne(test_feats, test_y, tsne_png, seed=args.seed)
        print(f"Saved t-SNE to {tsne_png}")

    # Explain a few test sequences
    idxs = np.random.RandomState(args.seed).choice(len(ds["test"]), size=min(args.explain_k, len(ds["test"])), replace=False)

    explanations = []
    for j, idx in enumerate(idxs.tolist()):
        ex = ds["test"][idx]
        seq = ex["sequence"]
        lab = int(ex["label"])

        res = importance_random_drop(
            backbone, probe, tokenizer, seq, lab,
            device=args.device, max_len=args.max_len,
            mc_samples=args.mc_samples, drop_prob=args.drop_prob, seed=args.seed + 1337 + j,
        )

        # Save JSON
        out_json = os.path.join(args.outdir, f"explain_{args.task}_{idx}.json")
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)

        # Plot
        title = (f"{args.task} test idx={idx} label={lab} "
                 f"baseline p={np.round(res['baseline_probs'], 3).tolist()} "
                 f"H={res['baseline_entropy']:.3f}")
        out_png = os.path.join(args.outdir, f"explain_{args.task}_{idx}.png")
        plot_sequence_importance(
            res["nuc_seq"], res["nuc_importance"], out_png, title, max_show=args.explain_max_show
        )

        print(f"Saved explanation: {out_png}")
        explanations.append({"idx": idx, "json": out_json, "png": out_png})

    # Write a tiny manifest for convenience
    manifest_path = os.path.join(args.outdir, f"manifest_{args.task}.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {"ckpt": args.ckpt, "task": args.task, "acc": float(acc), "probe": probe_path, "examples": explanations},
            f, indent=2
        )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()