import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb, hydra, tqdm, random, math
import pyfaidx, pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast  # Updated import

# --- 1. RoPE & Transformer Components ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # Ensure dim is even for rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        # Create cos/sin cache: (MaxLen, Dim/2) -> (MaxLen, Dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None, positions=None):
        # x: (B, H, L, D)
        # positions: (B, L) -> integer indices of original positions
        if positions is None:
            return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
        
        # Gather frequencies based on positions: (B, L, D)
        cos = self.cos_cached[positions].unsqueeze(1) # (B, 1, L, D)
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
        
        # Explicit Projections for safe RoPE injection
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        
        # 1. Attention with RoPE
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Inject RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Scaled Dot Product Attention
        # Note: is_causal=False for bidirectional DNA modeling
        out = F.scaled_dot_product_attention(q, k, v) 
        
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        x = residual + out
        
        # 2. MLP
        x = x + self.mlp(self.norm2(x))
        return x

class DNAEncoder(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.embed = nn.Embedding(4096 + 2, dim) # 4^6 vocab
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Fix: Dynamic head_dim for RoPE
        self.rope = RotaryEmbedding(self.head_dim) 
        
        self.blocks = nn.ModuleList([
            Block(dim, num_heads=num_heads) for _ in range(depth)
        ])
        
        # Projector
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, pos):
        # x: (B, L_subset)
        # pos: (B, L_subset)
        
        B, L = x.shape
        
        # 1. Embed
        x = self.embed(x) 
        
        # 2. CLS Token handling
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Adjust positions for CLS (at 0), shift data by 1
        cls_pos = torch.zeros((B, 1), device=pos.device, dtype=torch.long)
        pos = torch.cat((cls_pos, pos + 1), dim=1) 
        
        # 3. Get RoPE
        cos, sin = self.rope(x, positions=pos)
        
        # 4. Transformer
        for block in self.blocks:
            x = block(x, cos, sin)
            
        # 5. Pool
        cls_out = x[:, 0]
        return self.proj(cls_out)

# --- 2. Optimized Dataset ---
class DNADropDataset(Dataset):
    def __init__(self, fasta_path, intervals, seq_len=512, V=2, keep_ratio=0.5):
        self.fasta_path = fasta_path
        self.intervals = intervals
        self.seq_len = seq_len
        self.V = V
        self.keep_k = int(seq_len * keep_ratio)
        self.fasta = None
        self.k = 6
        
        # Pre-compute powers for vectorization
        self.powers = 4 ** torch.arange(self.k - 1, -1, -1)

    def _ensure_fasta(self):
        if self.fasta is None:
            self.fasta = pyfaidx.Fasta(self.fasta_path, sequence_always_upper=True)

    def _tokenize(self, seq_str):
        # OPTIMIZED: Vectorized tokenization (approx 50x faster)
        # Map string to bytes then tensors: A=65, C=67, G=71, T=84, N=78
        # We map A->0, C->1, G->2, T->3, N->4 (or anything else -> 4)
        
        # Create a tiny lookup table for ASCII
        # This is safe because seq_str is guaranteed uppercase by pyfaidx
        
        # ASCII values for ACGTN
        # A:65, C:67, G:71, T:84, N:78
        
        # Fast conversion to tensor indices
        seq_tensor = torch.tensor([ord(c) for c in seq_str], dtype=torch.long)
        
        # Map values: Default 4 (UNK)
        # Using a dense lookup for speed
        lookup = torch.full((128,), 4, dtype=torch.long)
        lookup[65] = 0 # A
        lookup[67] = 1 # C
        lookup[71] = 2 # G
        lookup[84] = 3 # T
        
        indices = lookup[seq_tensor] # (L,)
        
        # Sliding window using unfold (no loop!)
        # (L-k+1, k)
        windows = indices.unfold(0, self.k, 1) 
        
        # Check for UNKs in any window position
        # If any base is 4, the kmer is invalid
        has_unk = (windows == 4).any(dim=1)
        
        # Compute numerical tokens
        tokens = (windows * self.powers).sum(dim=1)
        
        # Mask UNKs
        tokens[has_unk] = 4096 
        
        return tokens

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        self._ensure_fasta()
        chrom, start, end = self.intervals[idx]
        
        try:
            seq_str = str(self.fasta[chrom][start:end])
        except:
            seq_str = "N" * self.seq_len
            
        if len(seq_str) < self.seq_len: seq_str += "N" * (self.seq_len - len(seq_str))
        seq_str = seq_str[:self.seq_len]
        
        full_tokens = self._tokenize(seq_str)
        
        # Ensure exact length logic same as before
        target_len = self.seq_len - self.k + 1
        if len(full_tokens) < target_len:
            full_tokens = F.pad(full_tokens, (0, target_len - len(full_tokens)), value=4096)
        full_tokens = full_tokens[:target_len]
        
        # Create Views
        views_tokens = []
        views_pos = []
        all_indices = torch.arange(len(full_tokens))
        
        for _ in range(self.V):
            perm = torch.randperm(len(full_tokens))
            keep_indices = perm[:self.keep_k]
            keep_indices, _ = keep_indices.sort() # Preserve order for RoPE
            
            views_tokens.append(full_tokens[keep_indices])
            views_pos.append(all_indices[keep_indices])
            
        return torch.stack(views_tokens), torch.stack(views_pos)

# --- 3. Utils (SIGReg, Intervals) ---
# Unchanged
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

def generate_intervals(fasta_path, stride=512):
    if not fasta_path.endswith('.fai'):
        pyfaidx.Fasta(fasta_path) 
    fai = pd.read_csv(fasta_path + '.fai', sep='\t', header=None, names=['c','l','x','y','z'])
    ints = []
    valid = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    for _, r in fai.iterrows():
        if r['c'] in valid:
            for i in range(0, r['l'] - stride, stride):
                ints.append((r['c'], i, i+stride))
    return ints

# --- 4. Main Loop with GradScaler ---
@hydra.main(version_base=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'): 
        cfg = OmegaConf.create({"bs": 128, "lr": 5e-4, "epochs": 10, "fasta": "hg38.fa", "lamb": 0.95})

    wandb.init(project="dna-jepa", config=dict(cfg))
    os.makedirs("checkpoints", exist_ok=True)
    
    intervals = generate_intervals(cfg.fasta)
    random.shuffle(intervals)
    # ds = DNADropDataset(cfg.fasta, intervals[:20000], seq_len=512, V=2, keep_ratio=0.6)
    ds = DNADropDataset(cfg.fasta, intervals, seq_len=512, V=2, keep_ratio=0.6)
    
    # Persistent workers for speed
    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    net = DNAEncoder(dim=512, depth=6, num_heads=8, proj_dim=128).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    
    # Initialize Scaler
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    print(f"Training on {len(ds)} samples...")
    
    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0.0

        for i, (toks, pos) in enumerate(tqdm.tqdm(loader)):
            B, V, K = toks.shape
            toks = toks.view(B*V, K).to("cuda")
            pos = pos.view(B*V, K).to("cuda")
            
            opt.zero_grad()
            
            # Use float16 for significant speedup + GradScaler usage
            with autocast("cuda", dtype=torch.float16):
                emb = net(toks, pos)
                proj = emb.view(B, V, -1)
                
                proj_mean = proj.mean(dim=1, keepdim=True)
                inv_loss = (proj - proj_mean).square().mean()
                reg_loss = sigreg(emb)
                
                loss = inv_loss * (1 - cfg.lamb) + reg_loss * cfg.lamb
            
            # Scaled Backward
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
            'optimizer_state_dict': opt.state_dict(),
            'loss': avg_loss,
            'config': dict(cfg)
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pth")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, "checkpoints/best_model.pth")
            
    torch.save(net.state_dict(), "checkpoints/final_model_weights.pth")

if __name__ == "__main__":
    main()
