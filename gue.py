import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_names
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio
import tqdm
import argparse
import os
import json

# Ensure Plotly saves as PNG
pio.templates.default = "plotly_white"

# --- 1. Model Definitions (MATCHING TRAINING) ---
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
    """
    embeddings: (N, D) numpy array
    labels: (N,) numpy array
    task_name: string
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run t-SNE
    print(f"  Computing t-SNE for {len(embeddings)} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(embeddings)
    
    # 2. Create Plotly Figure
    # Map numeric labels to text if binary
    label_text = labels.astype(str)
    if len(set(labels)) == 2:
        label_map = {'0': 'Negative (0)', '1': 'Positive (1)'}
        label_text = [label_map.get(l, l) for l in label_text]
        
    fig = px.scatter(
        x=projections[:, 0], 
        y=projections[:, 1],
        color=label_text,
        title=f"t-SNE Projection: {task_name}",
        labels={'color': 'Class'},
        opacity=0.7
    )
    
    # 3. Style Improvements
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="t-SNE Dim 1",
        yaxis_title="t-SNE Dim 2",
        template="plotly_white",
        width=1000,
        height=800
    )
    fig.update_traces(marker=dict(size=6))
    
    # 4. Save
    filename = os.path.join(output_dir, f"tsne_{task_name}.png")
    fig.write_image(filename)
    
    # Metadata for accessibility/tracking
    meta = {
        "caption": f"t-SNE visualization of DNA-JEPA embeddings for {task_name}",
        "description": "Scatter plot showing the separation of genomic sequence classes in the model's latent space."
    }
    with open(filename + ".meta.json", "w") as f:
        json.dump(meta, f)
        
    print(f"  Saved t-SNE plot to {filename}")

# --- 4. Evaluation & Visualization Logic ---
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

    # Collate
    def collate_fn(batch):
        seqs = [b[seq_key] for b in batch]
        labels = [int(b[label_key]) for b in batch]
        toks_list, pos_list = [], []
        for s in seqs:
            t = tokenizer.tokenize(s)
            if len(t) > 512: t = t[:512]
            elif len(t) < 512: t = torch.cat([t, torch.full((512 - len(t),), 4096)])
            toks_list.append(t)
            pos_list.append(torch.arange(len(t)))
        return torch.stack(toks_list), torch.stack(pos_list), torch.tensor(labels)

    train_dl = DataLoader(ds['train'], batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_split = 'test' if 'test' in ds else 'validation'
    test_dl = DataLoader(ds[test_split], batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 1. Train Linear Probe (Quick)
    probe = LinearProbe(512, num_classes).to("cuda")
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    backbone.eval()
    for epoch in range(2): # 2 epochs enough for probe usually
        probe.train()
        for toks, pos, y in tqdm.tqdm(train_dl, desc=f"  Ep {epoch}", leave=False):
            toks, pos, y = toks.to("cuda"), pos.to("cuda"), y.to("cuda")
            with torch.no_grad():
                feats = backbone(toks, pos, return_feats=True)
            loss = crit(probe(feats), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 2. Extract Embeddings for t-SNE (Sampled)
    # We use the test set for visualization to show generalization
    all_feats, all_labels = [], []
    sample_limit = 2000 # Max points for t-SNE
    
    probe.eval()
    with torch.no_grad():
        for toks, pos, y in test_dl:
            toks, pos = toks.to("cuda"), pos.to("cuda")
            feats = backbone(toks, pos, return_feats=True)
            
            # Collect for Metrics
            # (Skipping full metric calc loop for brevity, focusing on t-SNE extraction)
            
            # Collect for t-SNE
            if len(all_feats) < sample_limit:
                all_feats.append(feats.cpu().numpy())
                all_labels.append(y.numpy())
            else:
                break # collected enough
    
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
    configs = get_dataset_config_names("leannmlindsey/GUE")
    
    # Run first 5 tasks as demo (or remove slice for all)
    for config in configs[:5]: 
        eval_task(backbone, config, tokenizer)
