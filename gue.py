import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_names
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import argparse
import os
import json

# Ensure matplotlib backend is non-interactive for servers
import matplotlib
matplotlib.use('Agg') 

# --- 1. Model Definitions (UNCHANGED) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None, positions=None):
        if positions is None:
            return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)
        return cos, sin

def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return x + residual + self.mlp(self.norm2(out + residual))

class DNAEncoder(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.embed = nn.Embedding(4096 + 2, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.rope = RotaryEmbedding(self.head_dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads=num_heads) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, pos, return_feats=False):
        B, L = x.shape
        x = self.embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        cls_pos = torch.zeros((B, 1), device=pos.device, dtype=torch.long)
        pos = torch.cat((cls_pos, pos + 1), dim=1)
        cos, sin = self.rope(x, positions=pos)
        for block in self.blocks:
            x = block(x, cos, sin)
        cls_out = x[:, 0]
        if return_feats: return cls_out
        return self.proj(cls_out)

# --- 2. Utils ---
class KmerTokenizer:
    def __init__(self, k=6):
        self.k = k
        self.powers = 4 ** torch.arange(k - 1, -1, -1)

    def tokenize(self, seq_str):
        seq_tensor = torch.tensor([ord(c) for c in seq_str.upper()], dtype=torch.long)
        lookup = torch.full((128,), 4, dtype=torch.long)
        lookup[65], lookup[67], lookup[71], lookup[84] = 0, 1, 2, 3
        indices = lookup[seq_tensor]
        if len(indices) < self.k: return torch.tensor([4096], dtype=torch.long)
        windows = indices.unfold(0, self.k, 1)
        has_unk = (windows == 4).any(dim=1)
        tokens = (windows * self.powers).sum(dim=1)
        tokens[has_unk] = 4096
        return tokens

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x): return self.linear(self.bn(x))

# --- 3. Visualization Function ---
def create_tsne_plot(embeddings, labels, task_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Computing t-SNE for {len(embeddings)} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(embeddings)
    
    label_text = labels.astype(str)
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) == 2:
        mapping = {unique_labels[0]: 'Negative', unique_labels[1]: 'Positive'}
        hue_labels = [mapping[l] for l in labels]
    else:
        hue_labels = labels

    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    scatter = sns.scatterplot(
        x=projections[:, 0], 
        y=projections[:, 1],
        hue=hue_labels,
        palette="viridis",
        s=60,
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.title(f"t-SNE Projection: {task_name}", fontsize=16)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Class")
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"tsne_{task_name}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved t-SNE plot to {filename}")

# --- 4. Evaluation Logic ---
def eval_task(backbone, config_name, tokenizer):
    print(f"\n>>> Running GUE Task: {config_name}")
    
    try:
        ds = load_dataset("leannmlindsey/GUE", config_name, trust_remote_code=True)
    except Exception as e:
        print(f"Skipping {config_name}: Load failed ({e})")
        return None

    sample = ds['train'][0]
    keys = sample.keys()
    seq_key = next((k for k in ['sequence', 'seq', 'sentence', 'text'] if k in keys), None)
    label_key = next((k for k in ['label', 'targets', 'class'] if k in keys), None)
    
    if not seq_key or not label_key: return None

    try:
        num_classes = len(set(ds['train'][label_key]))
    except:
        num_classes = 2

    def collate_fn(batch):
        seqs = [b[seq_key] for b in batch]
        labels = [int(b[label_key]) for b in batch]
        toks_list, pos_list = [], []
        for s in seqs:
            t = tokenizer.tokenize(s)
            
            # --- VERIFICATION: NO MASKING ---
            # We are passing the tokens directly. 
            # If the sequence is longer than 512, we truncate (standard context window).
            # If it is shorter, we pad with 4096. 
            # We DO NOT apply any [MASK] token replacement here.
            
            if len(t) > 512: 
                t = t[:512]
            elif len(t) < 512: 
                t = torch.cat([t, torch.full((512 - len(t),), 4096)])
            
            toks_list.append(t)
            pos_list.append(torch.arange(len(t)))
            
        return torch.stack(toks_list), torch.stack(pos_list), torch.tensor(labels)

    train_dl = DataLoader(ds['train'], batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_split = 'test' if 'test' in ds else 'validation'
    test_dl = DataLoader(ds[test_split], batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 1. Train Linear Probe
    probe = LinearProbe(512, num_classes).to("cuda")
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    backbone.eval()
    for epoch in range(2): 
        probe.train()
        for toks, pos, y in tqdm.tqdm(train_dl, desc=f"  Ep {epoch}", leave=False):
            toks, pos, y = toks.to("cuda"), pos.to("cuda"), y.to("cuda")
            with torch.no_grad():
                # return_feats=True bypasses the projection head, giving us the raw cls_token (dim 512)
                feats = backbone(toks, pos, return_feats=True)
            loss = crit(probe(feats), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 2. Extract Embeddings (Sampled)
    all_feats, all_labels = [], []
    sample_limit = 2000 
    
    probe.eval()
    with torch.no_grad():
        for toks, pos, y in test_dl:
            toks, pos = toks.to("cuda"), pos.to("cuda")
            feats = backbone(toks, pos, return_feats=True)
            
            if len(all_feats) * 32 < sample_limit:
                all_feats.append(feats.cpu().numpy())
                all_labels.append(y.numpy())
            else:
                break 
    
    # 3. Generate t-SNE
    if all_feats:
        embeddings = np.concatenate(all_feats, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        create_tsne_plot(embeddings, labels, config_name)
        
    return "Done"

# --- 5. Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    backbone = DNAEncoder(dim=512, depth=6, num_heads=8, proj_dim=128).to("cuda")
    ckpt = torch.load(args.ckpt)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    backbone.load_state_dict({k.replace("module.", ""): v for k,v in state.items()})
    
    tokenizer = KmerTokenizer(k=6)
    
    # --- TASK SORTING LOGIC ---
    print("Fetching GUE configs...")
    configs = get_dataset_config_names("leannmlindsey/GUE")
    
    # Define keywords for human tasks (Promoters, Splice sites, Transcription Factors are usually human in GUE)
    # We prioritize tasks that explicitly mention 'human' or use standard human genomic feature names.
    human_keywords = ['human', 'prom', 'splice', 'tf']
    
    human_tasks = []
    other_tasks = []
    
    for c in configs:
        # Check if config matches human keywords AND is not explicitly mouse/yeast (unless it's TF which can be both)
        is_human = any(k in c.lower() for k in human_keywords)
        is_non_human = any(k in c.lower() for k in ['yeast', 'mouse', 'virus', 'covid', 'emp'])
        
        # Priority Logic: 
        # 1. If it has a human keyword and NOT a non-human keyword -> Definitely Human
        # 2. 'tf' is tricky (TF-M vs TF-H), but we will prioritize it generally.
        if is_human and not (is_non_human and 'tf' not in c.lower()):
            human_tasks.append(c)
        else:
            other_tasks.append(c)
            
    # Sort alphabetically for consistency
    human_tasks.sort()
    other_tasks.sort()
    
    # Combine: Human first
    sorted_configs = human_tasks + other_tasks
    
    print(f"Identified {len(human_tasks)} likely human tasks: {human_tasks}")
    print(f"Queued {len(sorted_configs)} total tasks.")

    # Run top 5 tasks (now prioritized by human)
    for config in sorted_configs[:5]: 
        eval_task(backbone, config, tokenizer)
