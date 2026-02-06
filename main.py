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

import torch._dynamo
import torch._inductor.config

torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

# --- Flash Attention Imports ---
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("❌ Flash Attention not installed. Run: pip install flash-attn --no-build-isolation")

# --- 1. Architecture Components (ALiBi + GEGLU + DNAEncoder) ---

class ALiBi(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        # Calculate slopes only [H]. Flash Attn handles the bias construction internally.
        slopes = torch.tensor([2 ** (-8 / num_heads * (i + 1)) for i in range(num_heads)])
        self.register_buffer("slopes", slopes)

    def forward(self):
        return self.slopes

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

    def forward(self, x, cu_seqlens, max_seqlen, alibi_slopes):
        """
        x: (total_nnz, dim) -> Flattened, unpadded input
        """
        residual = x
        x = self.norm1(x)
        
        # Reshape for Flash Attn: (total_nnz, n_heads, head_dim)
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Flash Attention Variable Length
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            alibi_slopes=alibi_slopes,
            causal=False
        )
        
        out = out.reshape(x.shape) # Flatten heads back to (total_nnz, dim)
        out = self.out_proj(out)
        
        # MLP handles flattened input natively
        return x + residual + self.mlp(self.norm2(out + residual))

class DNAEncoder(nn.Module):
    def __init__(self, tokenizer, dim=512, depth=6, num_heads=8, proj_dim=128):
        super().__init__()
        assert FLASH_AVAILABLE, "Flash Attention is required for this model version."
        self.dim = dim
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.embed = nn.Embedding(tokenizer.vocab_size, dim, padding_idx=self.pad_id)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.alibi = ALiBi(num_heads)
        self.blocks = nn.ModuleList([Block(dim, num_heads=num_heads) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, return_feats=False):
        B, L = x.shape
        
        # 1. Embed & Add CLS
        x_emb = self.embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1) # (B, L+1, D)
        
        # 2. Create Mask (True = Keep)
        # Combine CLS mask (always keep) with padding mask
        padding_mask = (x != self.pad_id)
        cls_mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
        mask = torch.cat((cls_mask, padding_mask), dim=1)
        
        # 3. Unpad for Flash Attn
        # x_unpad is (total_nnz, D)
        x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(x_emb, mask)
        
        # 4. Forward Blocks
        slopes = self.alibi()
        for block in self.blocks:
            x_unpad = block(x_unpad, cu_seqlens, max_seqlen, slopes)
            
        # 5. Repad to (B, L+1, D)
        x_out = pad_input(x_unpad, indices, B, L + 1)
        
        cls_out = x_out[:, 0]
        if return_feats: return cls_out
        return self.proj(cls_out)

# --- 2. Dataset & Losses (Unchanged) ---
import os
import glob
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import Dataset as TorchDataset


class DNATextDataset(TorchDataset):
    def __init__(self, txt_path, tokenizer, max_len=512):
        self.pad_id = tokenizer.pad_token_id or 0
        self.max_len = max_len
        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        base_name = os.path.basename(txt_path)
        
        # 1. Stable, merged cache path (Preferred)
        self.merged_path = os.path.join(self.cache_dir, f"tokenized_{max_len}_{base_name}_merged")
        
        # 2. Legacy sharded files pattern (Fallback)
        # Matches files like: tokenized_512_data.txt_00000_of_00016.arrow
        # Using specific glob pattern to find ANY existing shards
        shard_pattern = os.path.join(self.cache_dir, f"tokenized_{max_len}_*_of_*.arrow")

        # --- LOGIC FLOW ---
        
        # A. If stable merged dataset exists, load it (Fastest)
        if os.path.exists(self.merged_path):
            print(f"Loading merged dataset from {self.merged_path}")
            self.dataset = load_from_disk(self.merged_path)

        # B. If no merged dataset, but SHARDS exist -> Load & Merge
        elif glob.glob(shard_pattern):
            shards = sorted(glob.glob(shard_pattern))
            print(f"Found {len(shards)} existing sharded Arrow files. Merging...")
            
            # Use 'arrow' builder to load raw files without 'map' overhead
            dataset = load_dataset("arrow", data_files=shards, split="train", cache_dir="./cache")
            
            # Save immediately to lock in the merged version
            dataset.save_to_disk(self.merged_path)
            self.dataset = dataset
            print(f"Merged and saved to {self.merged_path}")

        # C. Nothing exists -> Compute from scratch
        else:
            print("No cache found. Processing...")
            dataset = load_dataset("text", data_files={"train": txt_path}, split="train", cache_dir=self.cache_dir)
            
            # Tokenize using max CPUs
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"], 
                    truncation=True, 
                    max_length=max_len, 
                    padding="max_length", 
                    add_special_tokens=False
                ),
                batched=True,
                num_proc=os.cpu_count(),
                remove_columns=["text"],
            )
            
            # Save the result so next time we hit Case A
            dataset.save_to_disk(self.merged_path)
            self.dataset = dataset

        # Zero-copy conversion for PyTorch
        self.dataset.set_format(type="torch", columns=["input_ids"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["input_ids"]






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

# --- 3. GUE Evaluation Logic (Unchanged) ---
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
            
            # --- FIX: Enable autocast for FlashAttention compatibility ---
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                feats = model(toks, return_feats=True)
            
            # Cast back to float32 for the Linear Probe (sklearn/torch stability)
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
            
            feat_dim = train_feats.shape[1]  # This will be 768
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
    
    # --- FIX: Sanitize keys from torch.compile artifacts ---
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip "_orig_mod." prefix if present
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
            
    # Load the sanitized state dict
    net.load_state_dict(new_state_dict)
    print(f"  ✓ Loaded model weights (fixed compiler prefixes)")
    
    if opt is not None and 'optimizer_state_dict' in checkpoint:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")
    
    start_step = checkpoint.get('step', 0)
    loaded_config = checkpoint.get('config', None)
    print(f"  ✓ Resuming from step {start_step}")
    return start_step, loaded_config


# --- 4. Main Training Loop (Unchanged) ---
@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'):
        cfg = OmegaConf.create({
            "bs": 128, "lr": 2e-4, "epochs": 50, 
            "train_file": "train.txt", "dev_file": "dev.txt",
            "lamb": 0.1, "V": 2, "proj_dim": 128, "keep_ratio": 0.7, "max_len": 512,
            "eval_every": 10000, 
            "resume_from": None,
        })
    
    wandb.init(project="dna-jepa", config=dict(cfg))
    os.makedirs("checkpoints", exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir="./")
    
    net = DNAEncoder(tokenizer, dim=512, depth=6, num_heads=8, proj_dim=cfg.proj_dim).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2, fused=True)

    start_step = 0
    if cfg.resume_from is not None:
        start_step, loaded_cfg = load_checkpoint(cfg.resume_from, net, opt, device="cuda")
        if loaded_cfg:
            print(f"  Previous config: {loaded_cfg}")

    net = torch.compile(net, mode='max-autotune')
    scaler = GradScaler('cuda')
    
    train_ds = DNATextDataset(cfg.train_file, tokenizer, cfg.max_len)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.bs, 
        shuffle=True, 
        num_workers=os.cpu_count(), 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4
    )
    
    global_step = start_step
    print(f"Starting training loop (Eval every {cfg.eval_every} steps)...")

    for epoch in range(cfg.epochs):
        net.train()
        for i, batch_tokens in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # 1. Move Clean Data to GPU
            # batch_tokens: (B, L)
            batch_tokens = batch_tokens.to("cuda", non_blocking=True)
            B, L = batch_tokens.shape
            
            # 2. Replicate for V views on GPU
            # (B, L) -> (B, V, L) -> (B*V, L)
            flat_input = batch_tokens.unsqueeze(1).expand(-1, cfg.V, -1).reshape(B * cfg.V, L)
            
            # 3. Create Mask on GPU
            # Create a random mask for every token
            rand_mask = torch.rand((B * cfg.V, L), device="cuda")
            
            # Calculate valid tokens (not padding)
            real_tokens_mask = (flat_input != tokenizer.pad_token_id)
            
            # Keep token IF: (Random < keep_ratio) AND (It is not a pad token)
            # This keeps the geometry of the sequence but "blanks out" tokens we don't want.
            # Since your DNAEncoder uses 'unpad_input', blanking them to PAD_ID removes them from attention.
            keep_mask = (rand_mask < cfg.keep_ratio) & real_tokens_mask
            
            # 4. Apply Mask
            aug_input = flat_input.clone()
            aug_input[~keep_mask] = tokenizer.pad_token_id

            
            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                emb = net(aug_input)
                proj = emb.view(B, cfg.V, -1).transpose(0, 1) # (V, B, D)
                
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
