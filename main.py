import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb, hydra, tqdm, random, math
import pyfaidx, pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer

# --- 1. RoPE & Transformer Components (UNCHANGED) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
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

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

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

    def forward(self, x, cos, sin, mask=None):
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Explicit attention mask handling for padding
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x

class DNAEncoder(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # DNABERT-2 vocab is ~4096 tokens + specials
        self.embed = nn.Embedding(4096 + 5, dim, padding_idx=0) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.rope = RotaryEmbedding(self.head_dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads=num_heads) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, pos):
        B, L = x.shape
        
        # Create padding mask (True where valid, False where padded)
        # Assuming 0 is padding index from tokenizer
        mask = (x != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        
        x = self.embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Adjust mask for CLS token (always attend to CLS)
        cls_mask = torch.ones((B, 1, 1, 1), device=x.device, dtype=torch.bool)
        mask = torch.cat((cls_mask, mask), dim=-1)
        
        cls_pos = torch.zeros((B, 1), device=pos.device, dtype=torch.long)
        pos = torch.cat((cls_pos, pos + 1), dim=1)
        
        cos, sin = self.rope(x, positions=pos)
        
        for block in self.blocks:
            x = block(x, cos, sin, mask=mask)
            
        cls_out = x[:, 0]
        return self.proj(cls_out)


# --- 2. BPE Optimized Dataset ---
class DNABPEDataset(Dataset):
    def __init__(self, fasta_path, intervals, max_len=512, V=2, keep_ratio=0.5):
        self.fasta_path = fasta_path
        self.intervals = intervals
        self.max_len = max_len
        self.V = V
        self.keep_ratio = keep_ratio
        self.fasta = None
        
        # Load official DNABERT-2 tokenizer
        print("Loading DNABERT-2 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir="./")
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def _ensure_fasta(self):
        if self.fasta is None:
            self.fasta = pyfaidx.Fasta(self.fasta_path, sequence_always_upper=True)

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        self._ensure_fasta()
        chrom, start, end = self.intervals[idx]
        
        # 1. Fetch raw sequence
        try:
            seq_str = str(self.fasta[chrom][start:end])
        except:
            seq_str = "N" * (end - start)
            
        # 2. BPE Tokenization
        # DNABERT-2 tokenizer handles raw strings directly
        tokens = self.tokenizer(seq_str, add_special_tokens=False)["input_ids"]
        tokens = torch.tensor(tokens, dtype=torch.long)

        # 3. Truncate/Pad to fixed window for processing
        # Note: BPE is variable length. 512 tokens covers MUCH more DNA than 512bp.
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # 4. Create Views (Subsampling Logic)
        views_tokens = []
        views_pos = []
        
        actual_len = len(tokens)
        keep_k = max(1, int(actual_len * self.keep_ratio))
        all_indices = torch.arange(actual_len)

        for _ in range(self.V):
            # Random drop
            perm = torch.randperm(actual_len)
            keep_indices = perm[:keep_k]
            keep_indices, _ = keep_indices.sort()
            
            sub_tokens = tokens[keep_indices]
            sub_pos = all_indices[keep_indices]
            
            # Pad to max_len for batching
            pad_len = self.max_len - len(sub_tokens)
            if pad_len > 0:
                sub_tokens = F.pad(sub_tokens, (0, pad_len), value=self.pad_id)
                sub_pos = F.pad(sub_pos, (0, pad_len), value=0) # Pos 0 is fine, will be masked
            
            views_tokens.append(sub_tokens)
            views_pos.append(sub_pos)
            
        return torch.stack(views_tokens), torch.stack(views_pos)

# --- 3. Utils (UNCHANGED) ---
class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
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

def generate_intervals(fasta_path, window_bp=2000, stride=1000):
    # Adjusted window_bp: BPE compresses ~4-5x. 
    # 512 tokens * 4bp/token ≈ 2000bp. We fetch larger chunks now.
    if not fasta_path.endswith('.fai'):
        pyfaidx.Fasta(fasta_path) 
    fai = pd.read_csv(fasta_path + '.fai', sep='\t', header=None, names=['c','l','x','y','z'])
    ints = []
    valid = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    for _, r in fai.iterrows():
        if r['c'] in valid:
            for i in range(0, r['l'] - window_bp, stride):
                ints.append((r['c'], i, i+window_bp))
    return ints

# --- 4. Main Loop (UNCHANGED logic, new params) ---
@hydra.main(version_base=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'): 
        cfg = OmegaConf.create({"bs": 128, "lr": 5e-4, "epochs": 10, "fasta": "hg38.fa", "lamb": 0.95})

    wandb.init(project="dna-jepa", config=dict(cfg))
    os.makedirs("checkpoints", exist_ok=True)
    
    # Generate larger intervals for BPE (2000bp instead of 512bp)
    intervals = generate_intervals(cfg.fasta, window_bp=2000, stride=1000)
    random.shuffle(intervals)
    
    # Use new BPE Dataset
    ds = DNABPEDataset(cfg.fasta, intervals[:100000], max_len=512, V=2, keep_ratio=0.8)
    
    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model now handles BPE vocab
    net = DNAEncoder(dim=512, depth=6, num_heads=8, proj_dim=128).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    print(f"Training on {len(ds)} samples with BPE...")
    
    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0.0

        for i, (toks, pos) in enumerate(tqdm.tqdm(loader)):
            B, V, K = toks.shape
            toks = toks.view(B*V, K).to("cuda")
            pos = pos.view(B*V, K).to("cuda")
            
            opt.zero_grad()
            
            with autocast("cuda", dtype=torch.float16):
                emb = net(toks, pos)
                proj = emb.view(B, V, -1)
                
                proj_mean = proj.mean(dim=1, keepdim=True)
                inv_loss = (proj - proj_mean).square().mean()
                reg_loss = sigreg(emb)
                
                loss = inv_loss * (1 - cfg.lamb) + reg_loss * cfg.lamb
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()

            if i % 10 == 0:
                wandb.log({"loss": loss.item(), "inv": inv_loss.item(), "reg": reg_loss.item()})
                
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pth")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, "checkpoints/best_model.pth")
            print(f"  * New best model saved! (Loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
