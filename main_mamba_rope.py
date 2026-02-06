import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import wandb, hydra, tqdm, random
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")


# --- 1. Architecture Components (RoPE + GEGLU + BiMamba) ---
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)
        
        # Precompute for max sequence length
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, theta)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Rotate: split x into pairs and apply rotation
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 : self.dim]
        
        # Apply rotation matrix
        rotated = torch.cat([
            x1 * cos[:, : self.dim // 2] - x2 * sin[:, : self.dim // 2],
            x2 * cos[:, self.dim // 2 :] + x1 * sin[:, self.dim // 2 :]
        ], dim=-1)
        
        return rotated


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BiMambaBlock(nn.Module):
    """
    Bi-Directional Mamba Block.
    Runs Mamba forward and backward (on flipped sequence) and combines outputs.
    This restores the global receptive field needed for an Encoder.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Two independent Mamba towers for Forward and Backward
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(GEGLU(dim, hidden_dim), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        # 1. Bi-Directional Mamba Step
        residual = x
        x = self.norm1(x)
        
        # Forward pass
        out_fwd = self.mamba_fwd(x)
        
        # Backward pass (flip seq, run mamba, flip back)
        out_bwd = self.mamba_bwd(x.flip((1,))).flip((1,))
        
        # Combine (Simple addition is standard for BiMamba)
        x = residual + (out_fwd + out_bwd)
        
        # 2. MLP Step
        x = x + self.mlp(self.norm2(x))
        return x


class DNAEncoder(nn.Module):
    def __init__(self, tokenizer, dim=512, depth=6, d_state=16, d_conv=4, expand=2, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.embed = nn.Embedding(tokenizer.vocab_size, dim, padding_idx=self.pad_id)
        self.rope = RoPE(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Swapped MambaBlock for BiMambaBlock
        self.blocks = nn.ModuleList([
            BiMambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand) 
            for _ in range(depth)
        ])
        
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, return_feats=False):
        B, L = x.shape
        x = self.embed(x)
        x = self.rope(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for block in self.blocks:
            x = block(x)
            
        cls_out = x[:, 0]
        if return_feats: return cls_out
        return self.proj(cls_out)


# --- 2. Dataset & Losses ---
class DNATextDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_len=512, V=2, keep_ratio=0.5):
        self.txt_path = txt_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.V = V
        self.keep_ratio = keep_ratio
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        print(f"Indexing {txt_path}...")
        self.line_offsets = []
        with open(txt_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
        print(f"Found {len(self.line_offsets)} sequences")

    def __len__(self): return len(self.line_offsets)

    def __getitem__(self, idx):
        with open(self.txt_path, 'r') as f:
            f.seek(self.line_offsets[idx])
            seq_str = f.readline().strip()
            
        if not seq_str or seq_str.count('N') / len(seq_str) > 0.1:
            return self.__getitem__((idx + 1) % len(self))
            
        tokens = self.tokenizer(seq_str, add_special_tokens=False)["input_ids"]
        tokens = torch.tensor(tokens, dtype=torch.long)
        if len(tokens) > self.max_len: tokens = tokens[:self.max_len]
        
        views = []
        # Optimization: Move probability check to vector operation
        # This removes the expensive .sort() call entirely
        
        for _ in range(self.V):
            # 1. Generate boolean mask (Fast, O(N))
            mask = torch.rand(len(tokens)) < self.keep_ratio
            
            # 2. Apply mask to preserve original order implicitly
            sub = tokens[mask]
            
            # Safety: Ensure we don't return an empty sequence
            if len(sub) == 0:
                # Fallback: pick one random token if mask removed everything
                sub = tokens[torch.randint(0, len(tokens), (1,))]

            # 3. Padding (Same as before)
            pad_len = self.max_len - len(sub)
            if pad_len > 0: 
                sub = F.pad(sub, (0, pad_len), value=self.pad_id)
            
            views.append(sub)
            
        return torch.stack(views)


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


# --- 3. GUE Evaluation Logic ---
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
            feats = model(toks, return_feats=True)
            all_feats.append(feats.cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def eval_gue(backbone, tokenizer, device="cuda", task_limit=3):
    """Run evaluation on a few GUE tasks."""
    print("\n>>> Running GUE Evaluation...")
    backbone.eval()
    
    # Pick a few fast tasks (Human Promoters, Splice, etc.)
    tasks = ['human_tf_0', 'prom_core_notata', 'splice_reconstructed'][:task_limit]
    results = {}
    
    for task in tasks:
        try:
            print(f"  Task: {task}")
            ds = load_dataset("leannmlindsey/GUE", task, cache_dir="./")
            
            # Simple collation for GUE (single view, no dropping)
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

            train_dl = DataLoader(ds['train'], batch_size=64, shuffle=False, collate_fn=collate_fn)
            test_dl = DataLoader(ds['test'], batch_size=64, shuffle=False, collate_fn=collate_fn)
            
            # 1. Extract Features
            train_feats, train_y = extract_features(backbone, train_dl, device)
            test_feats, test_y = extract_features(backbone, test_dl, device)
            
            # 2. Train Probe
            probe = LinearProbe(512, len(set(train_y.numpy()))).to(device)
            opt_probe = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            
            probe_ds = TensorDataset(train_feats.to(device), train_y.to(device))
            probe_dl = DataLoader(probe_ds, batch_size=64, shuffle=True)
            
            for _ in range(5): # 5 epochs for probe
                for fx, fy in probe_dl:
                    loss = crit(probe(fx), fy)
                    opt_probe.zero_grad()
                    loss.backward()
                    opt_probe.step()
            
            # 3. Test
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
    """
    Load a checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        net: The model instance
        opt: Optimizer instance (optional, for resuming training)
        device: Device to load tensors to
    
    Returns:
        Tuple of (start_step, loaded_config) or (0, None) if no checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0, None
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded model weights")
    
    # Load optimizer state if provided
    if opt is not None and 'optimizer_state_dict' in checkpoint:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")
    
    start_step = checkpoint.get('step', 0)
    loaded_config = checkpoint.get('config', None)
    
    print(f"  ✓ Resuming from step {start_step}")
    return start_step, loaded_config



# --- 4. Main Training Loop ---
@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'):
        cfg = OmegaConf.create({
            "bs": 32, "lr": 2e-4, "epochs": 50, 
            "train_file": "train.txt", "dev_file": "dev.txt",
            "lamb": 0.1, "V": 2, "proj_dim": 128, "keep_ratio": 0.7, "max_len": 512,
            "eval_every": 10000,  # Evaluate every 1000 steps
            "resume_from": None
        })
    
    wandb.init(project="dna-jepa", config=dict(cfg))
    os.makedirs("checkpoints", exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir="./")
    
    # 512 dim, d_state=16, d_conv=4 are standard Mamba defaults
    net = DNAEncoder(tokenizer, dim=512, depth=6, d_state=16, d_conv=4, expand=2, proj_dim=cfg.proj_dim).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2, fused=True)


    start_step = 0
    if cfg.resume_from is not None:
        start_step, loaded_cfg = load_checkpoint(cfg.resume_from, net, opt, device="cuda")
        if loaded_cfg:
            print(f"  Previous config: {loaded_cfg}")


    net = torch.compile(net, mode='max-autotune')
    scaler = GradScaler('cuda')
    
    train_ds = DNATextDataset(cfg.train_file, tokenizer, cfg.max_len, cfg.V, cfg.keep_ratio)
    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=4, pin_memory=True)
    
    global_step = start_step
    
    print(f"Starting training loop (Eval every {cfg.eval_every} steps)...")
    
    for epoch in range(cfg.epochs):
        net.train()
        for i, toks in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            B, V, K = toks.shape
            toks = toks.view(B * V, K).to("cuda", non_blocking=True)
            
            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                emb = net(toks)
                proj = emb.view(B, V, -1).transpose(0, 1) # (V, B, D)
                
                proj_mean = proj.mean(dim=0, keepdim=True)
                inv_loss = (proj - proj_mean).square().mean()
                reg_loss = sigreg(proj)
                loss = inv_loss * (1 - cfg.lamb) + reg_loss * cfg.lamb
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            wandb.log({"train/loss": loss.item(), "train/inv": inv_loss.item(), "train/reg": reg_loss.item(), "global_step": global_step})
            global_step += 1
            
            # --- Periodic Evaluation & Checkpointing ---
            if global_step % cfg.eval_every == 0:
                print(f"\n[Step {global_step}] Running Checkpoint & Evaluation...")
                
                # 1. Save Checkpoint
                ckpt_path = f"checkpoints/step_{global_step}.pth"
                torch.save({
                    'step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'config': dict(cfg)
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")
                
                # 2. Run GUE Eval
                gue_results = eval_gue(net, tokenizer, device="cuda")
                wandb.log({f"gue/{k}": v for k, v in gue_results.items()})
                
                net.train() # Switch back to train mode


if __name__ == "__main__":
    main()
