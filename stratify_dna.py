import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import load_dataset
import types
from pathlib import Path

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
            
        qkv = self.Wqkv(hidden_states) 
        
        try:
            qkv = qkv.reshape(qkv.shape[0], 3, num_heads, head_dim)
        except RuntimeError:
            if hasattr(self, "all_head_size"):
                 head_dim = self.all_head_size // num_heads
                 qkv = qkv.reshape(qkv.shape[0], 3, num_heads, head_dim)
            else:
                raise

        try:
            from flash_attn import flash_attn_varlen_qkvpacked_func
            
            if isinstance(cu_seqlens, torch.Tensor):
                 seqlens_in_batch = cu_seqlens[1:] - cu_seqlens[:-1]
                 max_seqlen_val = seqlens_in_batch.max().item()
            else:
                 max_seqlen_val = 512 

            out = flash_attn_varlen_qkvpacked_func(
                qkv, 
                cu_seqlens, 
                max_seqlen_val,
                dropout_p=0.0,
                softmax_scale=None, 
                causal=False
            )
            
            return out.reshape(out.shape[0], num_heads * head_dim)
            
        except ImportError:
            print("❌ flash_attn library not found. Please install it.")
            raise

    for layer in model.encoder.layer:
        layer.attention.self.forward = types.MethodType(patched_attention_forward, layer.attention.self)

def get_dnabert_embeddings(dataset, batch_size=32, max_len=None):
    from transformers import AutoTokenizer, AutoModel
    
    device = get_device()
    print(f"Using device: {device}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using precision: {dtype}")

    print("Loading DNABERT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR, torch_dtype=dtype)
    
    force_native_attention(model)
    
    model.to(device)
    model.eval()

    all_embeddings = []
    
    # Process the dataset (which is already sliced before passing here)
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
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    
    print(f"Generating {k}-mer profile embeddings...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    
    def seq_generator():
        for i in range(len(dataset)):
            yield dataset[i]["text"]
            
    print("Counting k-mers...")
    X = vectorizer.fit_transform(tqdm(seq_generator(), total=len(dataset), desc="K-mer Counting"))
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} k-mers")
    
    print("Normalizing to frequencies...")
    X = normalize(X, norm='l1', axis=1)
    
    return X

# -----------------------------------------------------------------------------
# 3. Main Logic & Plotting
# -----------------------------------------------------------------------------
def load_dataset_slice(file_path, chunk_id, chunk_size=1_000_000):
    """
    Loads the full dataset map but selects only the specific 1M slice.
    """
    print(f"Loading dataset index from {file_path}...")
    dataset = load_dataset(
        "text", 
        data_files={"train": file_path}, 
        split="train", 
        keep_in_memory=False, 
        cache_dir=CACHE_DIR
    )
    
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, len(dataset))
    
    print(f"Selecting slice: {start_idx} to {end_idx} (Size: {end_idx - start_idx})")
    
    if start_idx >= len(dataset):
        raise ValueError(f"Chunk ID {chunk_id} is out of range for dataset size {len(dataset)}")
        
    return dataset.select(range(start_idx, end_idx))

def plot_tsne(embeddings, labels, output_file="tsne.png"):
    """
    Runs t-SNE on the provided subset of embeddings.
    """
    print(f"Running t-SNE on {len(embeddings)} samples...")
    from scipy.sparse import issparse
    if issparse(embeddings):
        embeddings = embeddings.toarray()
    
    # Optional: Run PCA first to reduce to 50 dims if dims > 50 (Standard practice for speed)
    if embeddings.shape[1] > 50:
        print("Reducing dimensions with PCA (50 components) before t-SNE...")
        pca = PCA(n_components=50, random_state=42)
        embeddings = pca.fit_transform(embeddings)

    perp = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        projections[:, 0], 
        projections[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"t-SNE Projection (Subsampled 1/10)")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_file}")

def save_stratified_clusters(dataset, labels, chunk_id, output_dir="./stratification"):
    """
    Saves sequences into individual files based on cluster assignment.
    Format: ./stratification/{chunk_id}m_cluster{cluster_id}.txt
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving stratified sequences to {output_dir}...")
    
    # Organize indices by cluster to minimize file open/close operations
    cluster_map = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)
    
    # Write files
    for cluster_id, indices in tqdm(cluster_map.items(), desc="Writing Cluster Files"):
        filename = output_path / f"{chunk_id}m_cluster{cluster_id}.txt"
        
        with open(filename, "w") as f:
            for idx in indices:
                # dataset[idx] accesses the row in the slice
                seq = dataset[idx]["text"]
                f.write(seq + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input raw text file")
    parser.add_argument("--chunk_id", type=int, required=True, help="Chunk ID (0-32) indicating which million to process")
    parser.add_argument("--method", type=str, default="dnabert", choices=["dnabert", "kmer"], help="Embedding method")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters (default 100)")
    parser.add_argument("--k", type=int, default=4, help="K-mer size (only for --method kmer)")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--max_len", type=int, default=None, help="Max seq len")
    
    args = parser.parse_args()

    # 1. Load Data Slice
    dataset_slice = load_dataset_slice(args.input, args.chunk_id)
    
    # 2. Extract Embeddings
    if args.method == "dnabert":
        embeddings = get_dnabert_embeddings(dataset_slice, batch_size=args.batch_size, max_len=args.max_len)
    else:
        embeddings = get_kmer_embeddings(dataset_slice, k=args.k)
    
    # 3. Clustering
    print(f"Clustering into {args.n_clusters} strata using KMeans (High Iterations)...")
    # Increased max_iter for stability as requested
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10, max_iter=10000)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 4. Plotting (Deterministic 1/10 Sampling)
    # We take every 10th element starting from 0
    indices = np.arange(0, len(embeddings), 10)
    
    if len(indices) > 0:
        subset_embeddings = embeddings[indices]
        subset_labels = cluster_labels[indices]
        
        plot_name = f"./stratification/{args.chunk_id}m_tsne.png"
        # Ensure dir exists for plot
        os.makedirs(os.path.dirname(plot_name), exist_ok=True)
        
        plot_tsne(subset_embeddings, subset_labels, output_file=plot_name)

    # 5. Save Stratified Sequences
    save_stratified_clusters(dataset_slice, cluster_labels, args.chunk_id)

if __name__ == "__main__":
    main()
