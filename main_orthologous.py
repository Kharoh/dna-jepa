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
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import torch._dynamo
import torch._inductor.config

torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- GenomeMSA Import ---
try:
    from gpn.data import GenomeMSA
    GENOMEMSA_AVAILABLE = True
    print("✓ GenomeMSA available for ortholog augmentation")
except ImportError:
    GENOMEMSA_AVAILABLE = False
    print("⚠ GenomeMSA not available. Ortholog augmentation disabled.")

# --- Flash Attention Imports ---
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("❌ Flash Attention not installed. Run: pip install flash-attn --no-build-isolation")


# --- 1. Architecture Components (Same as before) ---

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
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary(x, cos_tok, sin_tok):
    head_dim = x.shape[-1]
    half = head_dim // 2
    
    x1 = x[..., :half]
    x2 = x[..., half:]
    
    c1 = cos_tok[..., :half]
    c2 = cos_tok[..., half:]
    s1 = sin_tok[..., :half] 
    s2 = sin_tok[..., half:]
    
    x_rot1 = x1 * c1 - x2 * s1
    x_rot2 = x1 * s2 + x2 * c1
    
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
        residual = x
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

        hidden_states = residual + out

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
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x, return_feats=False):
        B, L = x.shape
        x_emb = self.embed(x)

        padding_mask = (x != self.pad_id)

        x_unpad, indices, cu_seqlens, max_seqlen_compressed, seqlens = unpad_input(x_emb, padding_mask)

        seq_pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_idx = seq_pos[padding_mask] 

        rope_cos_l, rope_sin_l = self.rotary(L, device=x_unpad.device)

        for block in self.blocks:
            x_unpad = block(x_unpad, cu_seqlens, max_seqlen_compressed, pos_idx, rope_cos_l, rope_sin_l)

        x_out = pad_input(x_unpad, indices, B, L)

        mask_float = padding_mask.unsqueeze(-1).to(dtype=x_out.dtype)
        sum_embeddings = (x_out * mask_float).sum(dim=1)
        sum_counts = mask_float.sum(dim=1).clamp(min=1e-9)
        global_rep = sum_embeddings / sum_counts

        if return_feats:
            return global_rep
            
        return self.proj(global_rep)


# --- 2. Parallel Orthologous Augmentation Helper ---

class ParallelOrthologHelper:
    """
    Helper class with parallelized MSA fetching.
    Uses ThreadPoolExecutor to parallelize both:
    1. Multiple window queries within a single fetch
    2. Multiple fetch attempts simultaneously
    """
    
    def __init__(
        self,
        bed_file,
        msa_path="zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip",
        window_size=90,
        min_alignment_quality=0.25,
        max_workers=4,  # Number of parallel threads
    ):
        self.window_size = window_size
        self.min_alignment_quality = min_alignment_quality
        self.max_workers = max_workers
        
        # Statistics
        self.stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_msa_queries': 0,
        }
        self.stats_lock = threading.Lock()
        
        print(f"Loading GenomeMSA from {msa_path}...")
        self.genome_msa = GenomeMSA(msa_path)
        
        print(f"Loading coordinates from {bed_file}...")
        self.coordinates = self._load_bed(bed_file)
        print(f"  ✓ Loaded {len(self.coordinates)} functional regions")
        
        self.close_species = list(range(1, 21))
        self.distant_species = list(range(21, 90))
        self.all_species = self.close_species + self.distant_species
        
        self.rc_map = {
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n', '-': '-'
        }
        
        # Thread pool for parallel MSA queries
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        print(f"  ✓ Parallel executor initialized ({self.max_workers} workers)")
    
    def _load_bed(self, bed_file):
        """Load genomic coordinates from BED file."""
        coords = []
        with open(bed_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                chrom = parts[0]
                if chrom.startswith('chr'):
                    chrom = chrom[3:]
                
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                
                strand = '+' if len(parts) < 6 else parts[5]
                
                coords.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand
                })
        return coords
    
    def _array_to_seq(self, arr):
        """Convert GenomeMSA array to DNA string."""
        if isinstance(arr, np.ndarray):
            if arr.dtype.kind in ['U', 'S', 'O']:
                result = []
                for x in arr:
                    if isinstance(x, bytes):
                        result.append(x.decode())
                    else:
                        result.append(str(x))
                return ''.join(result)
            elif arr.dtype.kind in ['i', 'u']:
                token_map = {0: '-', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}
                return ''.join(token_map.get(int(t), 'N') for t in arr)
            else:
                return ''.join(str(x) for x in arr)
        elif isinstance(arr, str):
            return arr
        elif isinstance(arr, bytes):
            return arr.decode()
        else:
            return str(arr)
    
    def _fetch_msa_window(self, coord, chunk_start, chunk_end):
        """Fetch a single MSA window (to be parallelized)."""
        try:
            with self.stats_lock:
                self.stats['total_msa_queries'] += 1
            
            msa = self.genome_msa.get_msa(
                chrom=coord['chrom'],
                start=chunk_start,
                end=chunk_end,
                strand=coord['strand'],
                tokenize=False
            )
            
            return {
                'msa': msa,
                'start': chunk_start,
                'end': chunk_end,
                'success': True
            }
        except Exception as e:
            return {
                'msa': None,
                'start': chunk_start,
                'end': chunk_end,
                'success': False,
                'error': str(e)
            }
    
    def _try_single_fetch(self, max_len, species_preference):
        """
        Try to fetch one ortholog pair (to be parallelized).
        This is a single attempt that can run in parallel with other attempts.
        """
        # Pick random region
        coord = random.choice(self.coordinates)
        region_len = coord['end'] - coord['start']
        
        if region_len > max_len:
            offset = random.randint(0, region_len - max_len)
            query_start = coord['start'] + offset
            query_end = query_start + max_len
        else:
            query_start = coord['start']
            query_end = coord['end']
        
        try:
            # Create window queries
            window_ranges = []
            for chunk_start in range(query_start, query_end, self.window_size):
                chunk_end = min(chunk_start + self.window_size, query_end)
                window_ranges.append((chunk_start, chunk_end))
            
            # PARALLEL MSA QUERIES FOR ALL WINDOWS
            futures = []
            for chunk_start, chunk_end in window_ranges:
                future = self.executor.submit(
                    self._fetch_msa_window,
                    coord,
                    chunk_start,
                    chunk_end
                )
                futures.append(future)
            
            # Collect results as they complete
            window_results = []
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    window_results.append(result)
            
            # Sort by start position to maintain order
            window_results.sort(key=lambda x: x['start'])
            
            if not window_results:
                return None
            
            # Process windows to extract sequences
            all_human_seqs = []
            all_ortho_seqs = []
            selected_species = None
            
            for result in window_results:
                msa = result['msa']
                
                if len(msa) == 0:
                    continue
                
                # Get human sequence
                human_seq_str = self._array_to_seq(msa[0]).replace('-', '')
                all_human_seqs.append(human_seq_str)
                
                # Select species on first chunk
                if selected_species is None:
                    if species_preference == 'close':
                        species_pool = self.close_species
                    elif species_preference == 'distant':
                        species_pool = self.distant_species
                    else:
                        species_pool = self.all_species
                    
                    valid_species = []
                    for sp_idx in species_pool:
                        if sp_idx < len(msa):
                            sp_seq_str = self._array_to_seq(msa[sp_idx])
                            alignment_quality = 1.0 - (sp_seq_str.count('-') / len(sp_seq_str))
                            if alignment_quality >= self.min_alignment_quality:
                                valid_species.append((sp_idx, alignment_quality))
                    
                    if not valid_species:
                        return None
                    
                    valid_species.sort(key=lambda x: x[1], reverse=True)
                    selected_species = random.choice([s[0] for s in valid_species[:min(5, len(valid_species))]])
                
                # Get ortholog sequence
                if selected_species < len(msa):
                    ortho_seq_str = self._array_to_seq(msa[selected_species]).replace('-', '')
                    all_ortho_seqs.append(ortho_seq_str)
            
            # Concatenate chunks
            full_human = ''.join(all_human_seqs)
            full_ortho = ''.join(all_ortho_seqs)
            
            # Check if we got good sequences
            if len(full_human) >= 50 and len(full_ortho) >= 50:
                metadata = {
                    'coord': coord,
                    'species_idx': selected_species,
                    'human_len': len(full_human),
                    'ortho_len': len(full_ortho),
                }
                return (full_human, full_ortho, metadata)
            
            return None
            
        except Exception as e:
            return None
    
    def get_ortholog_sequence(self, max_len=512, species_preference='mixed', parallel_attempts=3):
        """
        Get ortholog sequence with parallel attempts.
        
        Args:
            max_len: Maximum sequence length
            species_preference: 'close', 'distant', or 'mixed'
            parallel_attempts: Number of parallel fetch attempts
        
        Returns:
            (human_seq, ortho_seq, metadata) or None
        """
        with self.stats_lock:
            self.stats['total_fetches'] += 1
        
        # PARALLEL FETCH ATTEMPTS
        # Submit multiple fetch attempts in parallel
        futures = []
        for _ in range(parallel_attempts):
            future = self.executor.submit(
                self._try_single_fetch,
                max_len,
                species_preference
            )
            futures.append(future)
        
        # Return the first successful result
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with self.stats_lock:
                    self.stats['successful_fetches'] += 1
                
                # Cancel remaining futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                
                return result
        
        # All attempts failed
        with self.stats_lock:
            self.stats['failed_fetches'] += 1
        
        return None
    
    def get_stats(self):
        """Get statistics."""
        with self.stats_lock:
            return self.stats.copy()
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# --- 3. Dataset ---

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

        print(f"Loading dataset from {txt_path}...")
        dataset = load_dataset("text", data_files={"train": txt_path}, split="train", keep_in_memory=False, cache_dir="./")
        self.lines = dataset  
        print(f"Loaded {len(self.lines)} sequences.")

    def _reverse_complement(self, text):
        text = text.strip()
        return ''.join(self.rc_map.get(base, base) for base in reversed(text))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]['text']
        return text


# --- 4. Async Ortholog Collator with Parallel Fetching ---

class AsyncOrthologCollator:
    """
    Collator with background thread for async ortholog generation.
    Uses ParallelOrthologHelper for parallel MSA fetching.
    """
    
    def __init__(
        self,
        tokenizer,
        ortholog_helper,
        max_len=512,
        num_views=2,
        mask_ratio=0.3,
        use_rc=True,
        cache_size=100,
        ortholog_frequency=10,
    ):
        self.tokenizer = tokenizer
        self.ortholog_helper = ortholog_helper
        self.max_len = max_len
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.use_rc = use_rc
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.ortholog_frequency = ortholog_frequency
        
        self.batch_counter = 0
        
        self.rc_map = {
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
        }
        
        # Cache queue
        self.cache_queue = queue.Queue(maxsize=cache_size)
        self.cache_enabled = ortholog_helper is not None
        self.stop_caching = threading.Event()
        
        # Cache statistics
        self.cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'fallbacks': 0,
        }
        self.stats_lock = threading.Lock()
        
        if self.cache_enabled:
            self.cache_thread = threading.Thread(target=self._cache_worker, daemon=True)
            self.cache_thread.start()
            print(f"✓ Async ortholog caching started (cache_size={cache_size}, frequency=1/{ortholog_frequency})")
            
            print(f"  Warming up cache...", end='', flush=True)
            while self.cache_queue.qsize() < min(10, cache_size):
                time.sleep(0.1)
            print(f" {self.cache_queue.qsize()} samples ready!")
    
    def _cache_worker(self):
        """Background thread that continuously generates ortholog samples."""
        while not self.stop_caching.is_set():
            try:
                views = self._make_views_ortholog()
                
                if views is not None:
                    try:
                        self.cache_queue.put(views, timeout=0.1)
                    except queue.Full:
                        time.sleep(0.5)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"  ⚠ Cache worker error: {e}")
                time.sleep(1.0)
    
    def _reverse_complement(self, text):
        text = text.strip()
        return ''.join(self.rc_map.get(base, base) for base in reversed(text))
    
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
    
    def _make_views_normal(self, text):
        """Standard views: human sequence + reverse complement."""
        views = []
        for v in range(self.num_views):
            if v == 1 and self.use_rc and random.random() < 0.5:
                src_text = self._reverse_complement(text)
            else:
                src_text = text

            ids = self._encode_no_dropout(src_text)
            ids = ids[:self.max_len]
            padding = [self.pad_id] * (self.max_len - len(ids))
            view_tokens = torch.tensor(ids + padding, dtype=torch.long)

            seq_len = min(len(ids), self.max_len)
            mask_content = self._get_block_mask(seq_len, mask_ratio=self.mask_ratio)
            full_mask = torch.ones(self.max_len, dtype=torch.bool)
            full_mask[:seq_len] = mask_content

            masked_view = view_tokens.clone()
            masked_view[~full_mask] = self.pad_id
            views.append(masked_view)

        return torch.stack(views)
    
    def _make_views_ortholog(self):
        """Ortholog views: different species for each view."""
        views = []
        
        for v in range(self.num_views):
            species_pref = 'close' if v % 2 == 0 else 'distant'
            
            result = self.ortholog_helper.get_ortholog_sequence(
                max_len=self.max_len,
                species_preference=species_pref,
                parallel_attempts=3  # Try 3 regions in parallel
            )
            
            if result is None:
                return None
            
            human_seq, ortho_seq, metadata = result
            src_text = human_seq if v == 0 else ortho_seq
            
            ids = self._encode_no_dropout(src_text)
            ids = ids[:self.max_len]
            padding = [self.pad_id] * (self.max_len - len(ids))
            view_tokens = torch.tensor(ids + padding, dtype=torch.long)

            seq_len = min(len(ids), self.max_len)
            mask_content = self._get_block_mask(seq_len, mask_ratio=self.mask_ratio)
            full_mask = torch.ones(self.max_len, dtype=torch.bool)
            full_mask[:seq_len] = mask_content

            masked_view = view_tokens.clone()
            masked_view[~full_mask] = self.pad_id
            views.append(masked_view)

        return torch.stack(views)
    
    def __call__(self, batch_texts):
        """Collate function with async ortholog fetching."""
        self.batch_counter += 1
        use_ortholog = (
            (self.batch_counter % self.ortholog_frequency == 0) and 
            self.cache_enabled and 
            GENOMEMSA_AVAILABLE
        )
        
        batch_views = []
        
        if use_ortholog:
            for _ in batch_texts:
                try:
                    views = self.cache_queue.get(timeout=0.1)
                    batch_views.append(views)
                    
                    with self.stats_lock:
                        self.cache_stats['cache_hits'] += 1
                        
                except queue.Empty:
                    views = self._make_views_normal(batch_texts[0])
                    batch_views.append(views)
                    
                    with self.stats_lock:
                        self.cache_stats['cache_misses'] += 1
                        self.cache_stats['fallbacks'] += 1
        else:
            for text in batch_texts:
                views = self._make_views_normal(text)
                batch_views.append(views)
        
        return torch.stack(batch_views)
    
    def get_stats(self):
        """Get cache statistics."""
        with self.stats_lock:
            stats = self.cache_stats.copy()
        
        stats['queue_size'] = self.cache_queue.qsize()
        
        if self.ortholog_helper:
            stats.update(self.ortholog_helper.get_stats())
        
        return stats
    
    def __del__(self):
        """Stop caching thread on cleanup."""
        if hasattr(self, 'stop_caching'):
            self.stop_caching.set()
            if hasattr(self, 'cache_thread'):
                self.cache_thread.join(timeout=1.0)


# --- 5. Losses ---

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


# --- 6. Evaluation (same as before) ---

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


# --- 7. Main Training Loop ---

@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    if not hasattr(cfg, 'bs'):
        cfg = OmegaConf.create({
            "bs": 128,
            "lr": 2e-4,
            "epochs": 500,
            "train_file": "train.txt",
            "bed_file": "./bed_files/functional_regions_combined.bed",
            "lamb": 0.1,
            "V": 2,
            "proj_dim": 64,
            "max_len": 512, 
            "resume_from": None,
            "model_dim": 512,
            "depth": 6,
            "heads": 8,
            "mask_ratio": 0.3,
            "steps_per_epoch": 2000,
            "eval_every": 10000,
            "use_ortholog": True,
            "min_alignment_quality": 0.25,
            "ortholog_cache_size": 5121,
            "ortholog_frequency": 10,
            "parallel_workers": 10,  # Number of parallel threads for MSA
        })

    wandb.init(project="dna-jepa-ortholog", config=dict(cfg))
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

    scaler = GradScaler('cuda')

    # Create dataset
    train_ds = DNAJEPADataset(
        cfg.train_file,
        tokenizer,
        cfg.max_len,
        num_views=cfg.V,
        mask_ratio=cfg.mask_ratio,
        use_rc=True
    )

    # Create PARALLEL ortholog helper
    ortholog_helper = None
    if cfg.use_ortholog and GENOMEMSA_AVAILABLE:
        try:
            ortholog_helper = ParallelOrthologHelper(
                bed_file=cfg.bed_file,
                min_alignment_quality=cfg.min_alignment_quality,
                max_workers=cfg.parallel_workers  # PARALLEL WORKERS
            )
            print("✓ Parallel ortholog helper initialized")
        except Exception as e:
            print(f"⚠ Failed to initialize ortholog helper: {e}")
            print("  Continuing without ortholog augmentation")

    # Create async collate function
    collate_fn = AsyncOrthologCollator(
        tokenizer=tokenizer,
        ortholog_helper=ortholog_helper,
        max_len=cfg.max_len,
        num_views=cfg.V,
        mask_ratio=cfg.mask_ratio,
        use_rc=True,
        cache_size=cfg.ortholog_cache_size,
        ortholog_frequency=cfg.ortholog_frequency,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    global_step = start_step
    print(f"Starting training loop (Eval every {cfg.eval_every} steps)...")

    for epoch in range(cfg.epochs):
        net.train()
        
        for i, batch_views in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if i >= cfg.steps_per_epoch:
                break

            batch_views = batch_views.to("cuda", non_blocking=True)
            B, V, L = batch_views.shape

            flat_input = batch_views.reshape(B * V, L)

            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                emb = net(flat_input)
                proj = emb.view(B, V, -1).transpose(0, 1)

                proj_mean = proj.mean(dim=0, keepdim=True)
                inv_loss = (proj - proj_mean).square().mean()
                reg_loss = sigreg(proj)
                loss = inv_loss * (1 - cfg.lamb) + reg_loss * cfg.lamb

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            wandb.log({
                "train/loss": loss.item(),
                "train/inv": inv_loss.item(),
                "train/reg": reg_loss.item()
            }, step=global_step)
            
            if global_step % 100 == 0:
                stats = collate_fn.get_stats()
                wandb.log({
                    "ortholog/cache_hits": stats.get('cache_hits', 0),
                    "ortholog/cache_misses": stats.get('cache_misses', 0),
                    "ortholog/fallbacks": stats.get('fallbacks', 0),
                    "ortholog/queue_size": stats.get('queue_size', 0),
                    "ortholog/total_fetches": stats.get('total_fetches', 0),
                    "ortholog/total_msa_queries": stats.get('total_msa_queries', 0),
                    "ortholog/success_rate": stats.get('successful_fetches', 0) / max(stats.get('total_fetches', 1), 1),
                }, step=global_step)
            
            global_step += 1

            if global_step % cfg.eval_every == 0:
                print(f"\n[Step {global_step}] Running Checkpoint & Evaluation...")
                
                stats = collate_fn.get_stats()
                print(f"\nOrtholog Statistics:")
                print(f"  Cache hits: {stats.get('cache_hits', 0)}")
                print(f"  Cache misses: {stats.get('cache_misses', 0)}")
                print(f"  Queue size: {stats.get('queue_size', 0)}")
                print(f"  Total fetches: {stats.get('total_fetches', 0)}")
                print(f"  Total MSA queries: {stats.get('total_msa_queries', 0)}")
                print(f"  Success rate: {stats.get('successful_fetches', 0) / max(stats.get('total_fetches', 1), 1):.1%}\n")
                
                ckpt_path = f"checkpoints/step_{global_step}.pth"
                torch.save({
                    'step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'config': dict(cfg)
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

                gue_results = eval_gue(net, tokenizer, device="cuda")
                wandb.log({f"gue/{k}": v for k, v in gue_results.items()}, step=global_step)

                net.train()


if __name__ == "__main__":
    main()
