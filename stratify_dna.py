import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from datasets import load_dataset
import types

# -----------------------------------------------------------------------------
# 0. Global Setup
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_DIR = "./"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# -----------------------------------------------------------------------------
# 1. DNABERT-2 Logic (With Flash Attention Patch)
# -----------------------------------------------------------------------------
MODEL_NAME = "zhihan1996/DNABERT-2-117M"

def force_native_attention(model):
    """
    Replaces the custom DNABERT-2 attention with the official flash_attn library.
    """
    print("✨ Patching model to use official Flash Attention...")
    
    def patched_attention_forward(self, hidden_states, cu_seqlens, seqlen, max_seqlen_in_batch, indices, attn_mask=None, bias=None):
        # 1. Get Dimensions
        if hasattr(self, "num_attention_heads"):
            num_heads = self.num_attention_heads
        elif hasattr(self, "num_heads"):
            num_heads = self.num_heads
        else:
             num_heads = 12 
        
        if hasattr(self, "attention_head_size"):
            head_dim = self.attention_head_size
        elif hasattr(self, "head_dim"):
            head_dim = self.head_dim
        else:
            head_dim = 64
            
        # 2. Project Q, K, V
        qkv = self.Wqkv(hidden_states) 
        
        # 3. Reshape for Flash Attention (total_tokens, 3, num_heads, head_dim)
        try:
            qkv = qkv.reshape(qkv.shape[0], 3, num_heads, head_dim)
        except RuntimeError:
            if hasattr(self, "all_head_size"):
                 head_dim = self.all_head_size // num_heads
                 qkv = qkv.reshape(qkv.shape[0], 3, num_heads, head_dim)
            else:
                raise

        # 4. Call Flash Attention
        try:
            from flash_attn import flash_attn_varlen_qkvpacked_func
            
            # Calculate correct max_seqlen from cu_seqlens
            if isinstance(cu_seqlens, torch.Tensor):
                 seqlens_in_batch = cu_seqlens[1:] - cu_seqlens[:-1]
                 max_seqlen_val = seqlens_in_batch.max().item()
            else:
                 max_seqlen_val = 512 

            # Output shape: (total_tokens, num_heads, head_dim)
            out = flash_attn_varlen_qkvpacked_func(
                qkv, 
                cu_seqlens, 
                max_seqlen_val,
                dropout_p=0.0,
                softmax_scale=None, 
                causal=False
            )
            
            # 5. Flatten heads: (total_tokens, num_heads, head_dim) -> (total_tokens, hidden_dim)
            return out.reshape(out.shape[0], num_heads * head_dim)
            
        except ImportError:
            print("❌ flash_attn library not found. Please install it.")
            raise

    # Apply patch
    for layer in model.encoder.layer:
        layer.attention.self.forward = types.MethodType(patched_attention_forward, layer.attention.self)

def get_dnabert_embeddings(dataset, batch_size=32, max_len=None):
    """Generates embeddings using the DNABERT-2 foundation model."""
    from transformers import AutoTokenizer, AutoModel
    
    device = get_device()
    print(f"Using device: {device}")

    # Flash Attention requires fp16 or bf16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using precision: {dtype}")

    print("Loading DNABERT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR, torch_dtype=dtype)
    
    # Apply Patch
    force_native_attention(model)
    
    model.to(device)
    model.eval()

    all_embeddings = []
    total_len = len(dataset)
    
    for i in tqdm(range(0, total_len, batch_size), desc="Extracting DNABERT Embeddings"):
        batch_data = dataset[i : i + batch_size]
        batch_seqs = batch_data["text"] 

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="longest",
            truncation=True if max_len else False,
            max_length=max_len
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(**inputs)
            
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

            # Mean Pooling
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            mask_expanded = attention_mask.to(dtype=hidden_states.dtype)
            masked_hidden = hidden_states * mask_expanded
            sum_embeddings = torch.sum(masked_hidden, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_embeddings.float().cpu().numpy())

    return np.vstack(all_embeddings)

# -----------------------------------------------------------------------------
# 2. K-mer Profile Logic
# -----------------------------------------------------------------------------
def get_kmer_embeddings(dataset, k=4):
    """
    Generates sparse embeddings based on normalized k-mer frequency profiles.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    
    print(f"Generating {k}-mer profile embeddings...")
    print("Initializing CountVectorizer (analyzer='char')...")
    
    # Use char analyzer with ngram_range to capture k-mers (e.g., 'ATC', 'GGT')
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    
    # Create a generator to stream data into the vectorizer without loading all RAM
    def seq_generator():
        for i in range(len(dataset)):
            yield dataset[i]["text"]
            
    # Fit and transform (returns a sparse matrix)
    print("Counting k-mers...")
    X = vectorizer.fit_transform(tqdm(seq_generator(), total=len(dataset), desc="K-mer Counting"))
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} k-mers")
    
    # Normalize (L1 norm) to get Frequencies instead of raw counts
    # This ensures sequence length doesn't dominate the clustering
    print("Normalizing to frequencies...")
    X = normalize(X, norm='l1', axis=1)
    
    return X

# -----------------------------------------------------------------------------
# 3. Main Logic & Plotting
# -----------------------------------------------------------------------------
def load_dataset_streaming(file_path):
    print(f"Loading dataset from {file_path}...")
    dataset = load_dataset(
        "text", 
        data_files={"train": file_path}, 
        split="train", 
        keep_in_memory=False, 
        cache_dir=CACHE_DIR
    )
    return dataset

def plot_tsne(embeddings, labels, output_file="tsne_smoke.png"):
    print("Running t-SNE...")
    # Handle sparse input for t-SNE
    from scipy.sparse import issparse
    
    # If using sparse k-mers with large N, t-SNE might need TruncatedSVD first
    # But for smoke mode (1000 items), dense conversion is fine
    if issparse(embeddings):
        embeddings = embeddings.toarray()
        
    perp = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        projections[:, 0], 
        projections[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"t-SNE of DNA Embeddings (N={len(embeddings)})")
    plt.savefig(output_file, dpi=300)
    print(f"t-SNE plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input raw text file")
    parser.add_argument("--output", type=str, default="stratified_data.csv", help="Output CSV path")
    parser.add_argument("--method", type=str, default="dnabert", choices=["dnabert", "kmer"], help="Embedding method")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--k", type=int, default=4, help="K-mer size (only for --method kmer)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size (dnabert only)")
    parser.add_argument("--max_len", type=int, default=None, help="Max seq len (dnabert only)")
    parser.add_argument("--smoke", action="store_true", help="Run on first 1000 sequences only")
    
    args = parser.parse_args()

    # 1. Load Data
    dataset = load_dataset_streaming(args.input)

    # 2. Smoke Mode Limit
    if args.smoke:
        print("\n=== SMOKE MODE ===")
        dataset = dataset.select(range(min(1000, len(dataset))))

    print(f"Processing {len(dataset)} sequences...")

    # 3. Extract Embeddings
    if args.method == "dnabert":
        embeddings = get_dnabert_embeddings(dataset, batch_size=args.batch_size, max_len=args.max_len)
    else:
        embeddings = get_kmer_embeddings(dataset, k=args.k)
    
    # 4. Clustering
    print(f"Clustering into {args.n_clusters} strata using KMeans...")
    # KMeans works efficiently with both dense (numpy) and sparse (scipy) matrices
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 5. Save Results
    print(f"Saving to {args.output}...")
    import csv
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence", "cluster"])
        for idx in range(len(dataset)):
            seq = dataset[idx]["text"]
            writer.writerow([idx, seq, cluster_labels[idx]])

    # 6. Plotting
    if args.smoke:
        plot_tsne(embeddings, cluster_labels, output_file=f"tsne_{args.method}.png")

if __name__ == "__main__":
    main()
