import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import wandb, hydra, tqdm, random
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np

import torch._dynamo
import torch._inductor.config

torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Flash Attention Imports ---
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("❌ Flash Attention not installed. Run: pip install flash-attn --no-build-isolation")

# --- 1. Architecture Components (RoPE + GEGLU + DNAEncoder) ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, seqlen, device=None):
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(seqlen, device=device, dtype=torch.float32)  # (L,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)            # (L, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                      # (L, dim)
        cos = emb.cos()                                              # (L, dim)
        sin = emb.sin()                                              # (L, dim)
        return cos, sin                                              # no batch/head dims here


def apply_rotary(x, cos_tok, sin_tok):
    # Match the "cat(freqs, freqs)" construction format
    head_dim = x.shape[-1]
    half = head_dim // 2
    
    # Split input
    x1 = x[..., :half]
    x2 = x[..., half:]
    
    # Split RoPE cache 
    c1 = cos_tok[..., :half]
    c2 = cos_tok[..., half:]
    s1 = sin_tok[..., :half] 
    s2 = sin_tok[..., half:] # Already contains the second half of sines
    
    # Rotation
    x_rot1 = x1 * c1 - x2 * s1
    x_rot2 = x1 * s2 + x2 * c1 # Note: s2 (second half) is used here
    
    return torch.cat((x_rot1, x_rot2), dim=-1)



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

    def forward(self, x, cu_seqlens, max_seqlen, pos_idx, rope_cos_l, rope_sin_l):
        # 1. Save Residual (Input)
        residual = x
        
        # 2. Apply Norm and calculate Attention
        x = self.norm1(x)

        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        cos_tok = rope_cos_l[pos_idx].unsqueeze(1)
        sin_tok = rope_sin_l[pos_idx].unsqueeze(1)

        q = apply_rotary(q, cos_tok, sin_tok)
        k = apply_rotary(k, cos_tok, sin_tok)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False
        )

        out = out.reshape(x.shape)
        out = self.out_proj(out)

        # 3. First Residual Connection (Add Attention Output to Input)
        # Your previous code added 'x' (the normalized input) here by mistake.
        hidden_states = residual + out

        # 4. Second Residual Connection (Add MLP Output)
        # Apply Norm2 to the updated states, pass through MLP, and add back
        return hidden_states + self.mlp(self.norm2(hidden_states))



class DNAEncoder(nn.Module):
    def __init__(self, tokenizer, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        assert FLASH_AVAILABLE, "Flash Attention is required."
        self.dim = dim
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.embed = nn.Embedding(tokenizer.vocab_size, dim, padding_idx=self.pad_id)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rotary = RotaryEmbedding(self.head_dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads=num_heads) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LayerNorm(2048), # Changed from BatchNorm1d
            # nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048), # Changed from BatchNorm1d
            # nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, return_feats=False):
        B, L = x.shape
        # 1. Embed DNA tokens only (No CLS concatenation)
        x_emb = self.embed(x) # (B, L, D)

        padding_mask = (x != self.pad_id) # (B, L)

        # 2. Flash Attention Unpadding
        # unpad_input returns flattened valid tokens
        x_unpad, indices, cu_seqlens, max_seqlen_compressed, seqlens = unpad_input(x_emb, padding_mask)

        # 3. Construct Position Indices (Standard 0..N)
        # Since there is no CLS, we just map 0->0, 1->1, etc.
        seq_pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        
        # Flatten positions to match x_unpad
        pos_idx = seq_pos[padding_mask] 

        # 4. Generate RoPE Cache
        # Max index is L-1, so size L is sufficient
        rope_cos_l, rope_sin_l = self.rotary(L, device=x_unpad.device)

        # 5. Apply Transformer Blocks
        for block in self.blocks:
            x_unpad = block(x_unpad, cu_seqlens, max_seqlen_compressed, pos_idx, rope_cos_l, rope_sin_l)

        # 6. Repad sequences to (B, L, D)
        # Note: total_seqlen is L, not L+1
        x_out = pad_input(x_unpad, indices, B, L)

        # 7. Global Average Pooling (Masked)
        # We must ignore padding tokens in the average to avoid diluting the signal
        
        # Create float mask: (B, L, 1)
        mask_float = padding_mask.unsqueeze(-1).to(dtype=x_out.dtype)
        
        # Sum valid embeddings: (B, D)
        sum_embeddings = (x_out * mask_float).sum(dim=1)
        
        # Count valid tokens: (B, 1)
        # Clamp min=1e-9 to avoid division by zero if a sequence is fully padded (edge case)
        sum_counts = mask_float.sum(dim=1).clamp(min=1e-9)
        
        # Average
        global_rep = sum_embeddings / sum_counts

        if return_feats:
            return global_rep
            
        return self.proj(global_rep)




# --- 2. Dataset (no BPE Dropout, keep block masking) ---

class DNAJEPADataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_len=512, num_views=2, mask_ratio=0.3, use_rc=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.use_rc = use_rc and (num_views >= 2)

        self.rc_map = {
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
        }

        print(f"Loading dataset from {txt_path} using Memory Mapping...")
        # Use load_dataset to memory-map the file instead of reading into RAM
        # keep_in_memory=False ensures it stays on disk
        dataset = load_dataset("text", data_files={"train": txt_path}, split="train", keep_in_memory=False, cache_dir="./")
        self.lines = dataset  
        print(f"Loaded {len(self.lines)} sequences.")

    def _reverse_complement(self, text):
        """Return reverse complement of a DNA string."""
        text = text.strip()
        return ''.join(self.rc_map.get(base, base) for base in reversed(text))

    def __len__(self):
        return len(self.lines)

    def _encode_no_dropout(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len
        )

    def _get_block_mask(self, length, mask_ratio=0.3):
        mask = torch.ones(length, dtype=torch.bool)
        if length < 5:
            return mask

        num_to_mask = int(length * mask_ratio)
        num_masked = 0
        avg_block = max(3, length // 8)

        attempts = 0
        while num_masked < num_to_mask and attempts < 20:
            block_size = random.randint(avg_block // 2, avg_block * 2)
            if num_masked + block_size > num_to_mask:
                block_size = num_to_mask - num_masked

            start = random.randint(0, max(0, length - block_size))

            if not mask[start]:
                attempts += 1
                continue

            mask[start:start+block_size] = False
            num_masked += block_size

        return mask

    def __getitem__(self, idx):
        text = self.lines[idx]['text']

        # Generate V different views (same tokenization, different masks)
        views = []
        for v in range(self.num_views):
            if v == 1 and self.use_rc and random.random() < 0.5: # Apply reverse complement to the second view with 50% chance
                src_text = self._reverse_complement(text)
            else:
                src_text = text

            ids = self._encode_no_dropout(src_text)
            ids = ids[:self.max_len]
            padding = [self.tokenizer.pad_token_id] * (self.max_len - len(ids))
            view_tokens = torch.tensor(ids + padding, dtype=torch.long)

            seq_len = min(len(ids), self.max_len)
            mask_content = self._get_block_mask(seq_len, mask_ratio=self.mask_ratio)
            full_mask = torch.ones(self.max_len, dtype=torch.bool)
            full_mask[:seq_len] = mask_content

            masked_view = view_tokens.clone()
            masked_view[~full_mask] = self.tokenizer.pad_token_id

            views.append(masked_view)

        return torch.stack(views)

# --- 3. Losses ---

class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()

# --- 4. Evaluation ---

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x): return self.linear(self.bn(x))

def extract_features(model, loader, device="cuda"):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for toks, y in loader:
            toks = toks.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                feats = model(toks, return_feats=True)
            all_feats.append(feats.float().cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)

def eval_gue(backbone, tokenizer, device="cuda", task_limit=3):
    print("\n>>> Running GUE Evaluation...")
    backbone.eval()
    tasks = ['human_tf_0', 'prom_core_notata', 'splice_reconstructed'][:task_limit]
    results = {}

    for task in tasks:
        try:
            print(f"  Task: {task}")
            ds = load_dataset("leannmlindsey/GUE", task, cache_dir="./")

            def collate_fn(batch):
                seqs = [b['sequence'] for b in batch]
                labels = [int(b['label']) for b in batch]
                toks_list = []
                for s in seqs:
                    t = torch.tensor(tokenizer(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
                    if len(t) > 512: t = t[:512]
                    else: t = F.pad(t, (0, 512-len(t)), value=tokenizer.pad_token_id)
                    toks_list.append(t)
                return torch.stack(toks_list), torch.tensor(labels)

            train_dl = DataLoader(ds['train'], batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)
            test_dl = DataLoader(ds['test'], batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

            train_feats, train_y = extract_features(backbone, train_dl, device)
            test_feats, test_y = extract_features(backbone, test_dl, device)

            feat_dim = train_feats.shape[1]
            num_classes = len(set(train_y.numpy()))
            probe = LinearProbe(feat_dim, num_classes).to(device)
            opt_probe = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()

            probe_ds = TensorDataset(train_feats.to(device), train_y.to(device))
            probe_dl = DataLoader(probe_ds, batch_size=64, shuffle=True)

            for _ in range(5):
                for fx, fy in probe_dl:
                    loss = crit(probe(fx), fy)
                    opt_probe.zero_grad()
                    loss.backward()
                    opt_probe.step()

            probe.eval()
            with torch.no_grad():
                preds = probe(test_feats.to(device)).argmax(1).cpu()
                test_y_np = test_y.numpy()
                acc = accuracy_score(test_y_np, preds)
                mcc = matthews_corrcoef(test_y_np, preds)
                results[f"{task}_acc"] = acc
                results[f"{task}_mcc"] = mcc
                print(f"    Acc: {acc:.4f} | MCC: {mcc:.4f}")

        except Exception as e:
            print(f"    Failed: {e}")

    return results

def load_checkpoint(checkpoint_path, net, opt=None, device="cuda"):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0, None

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=False)
    print(f"  ✓ Loaded model weights")

    if opt is not None and 'optimizer_state_dict' in checkpoint:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")

    start_step = checkpoint.get('step', 0)
    loaded_config = checkpoint.get('config', None)
    return start_step, loaded_config

# --- 5. Main Training Loop ---

@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'):
        cfg = OmegaConf.create({
            "bs": 128, "lr": 2e-4, "epochs": 50,
            "train_file": "train.txt", "dev_file": "dev.txt",
            "lamb": 0.1, "V": 2, "proj_dim": 64,
            "max_len": 512, "eval_every": 10000,
            # "resume_from": "./checkpoints/step_30000.pth",
            "resume_from": None,
            "model_dim": 512,
            "depth": 6,
            "heads": 8,
            "mask_ratio": 0.3
        })

    wandb.init(project="dna-jepa", config=dict(cfg))
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir="./")

    net = DNAEncoder(
        tokenizer,
        dim=cfg.model_dim,
        depth=cfg.depth,
        num_heads=cfg.heads,
        proj_dim=cfg.proj_dim
    ).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2, fused=True)

    start_step = 0
    if cfg.resume_from is not None:
        start_step, loaded_cfg = load_checkpoint(cfg.resume_from, net, opt, device="cuda")

    # net = torch.compile(net, mode='max-autotune')
    scaler = GradScaler('cuda')

    train_ds = DNAJEPADataset(
        cfg.train_file,
        tokenizer,
        cfg.max_len,
        num_views=cfg.V,
        mask_ratio=cfg.mask_ratio,
        use_rc=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    global_step = start_step
    print(f"Starting training loop (Eval every {cfg.eval_every} steps)...")

    for epoch in range(cfg.epochs):
        net.train()
        for i, batch_views in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # batch_views: (B, V, L)
            batch_views = batch_views.to("cuda", non_blocking=True)
            B, V, L = batch_views.shape

            # Flatten to (B*V, L)
            flat_input = batch_views.reshape(B * V, L)

            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                emb = net(flat_input)  # (B*V, proj_dim)
                proj = emb.view(B, V, -1).transpose(0, 1)  # (V, B, D)

                proj_mean = proj.mean(dim=0, keepdim=True)
                inv_loss = (proj - proj_mean).square().mean()
                reg_loss = sigreg(proj)
                loss = inv_loss * (1 - cfg.lamb) + reg_loss * cfg.lamb

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            wandb.log({"train/loss": loss.item(), "train/inv": inv_loss.item(), "train/reg": reg_loss.item()}, step=global_step)
            global_step += 1

            if global_step % cfg.eval_every == 0:
                print(f"\n[Step {global_step}] Running Checkpoint & Evaluation...")
                ckpt_path = f"checkpoints/step_{global_step}.pth"
                torch.save({
                    'step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'config': dict(cfg)
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

                gue_results = eval_gue(net, tokenizer, device="cuda")
                wandb.log({f"gue/{k}": v for k, v in gue_results.items()})

                net.train()

if __name__ == "__main__":
    main()
