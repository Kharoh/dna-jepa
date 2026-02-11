# test_ortholog_dataset.py
"""
Test script for OrthologousAugmentedDataset with streaming GenomeMSA.
Fixed to handle numpy array MSA output.
"""

import os
import sys
import torch
import random
import itertools
from pathlib import Path
from collections import defaultdict
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer
import numpy as np

# Try to import GenomeMSA
try:
    from gpn.data import GenomeMSA
    GENOMEMSA_AVAILABLE = True
    print("✓ GenomeMSA available")
except ImportError:
    GENOMEMSA_AVAILABLE = False
    print("❌ GenomeMSA not available. Install with: pip install git+https://github.com/songlab-cal/gpn.git")


class OrthologousAugmentedDataset(IterableDataset):
    """
    Streams orthologous sequences from remote multiz100way alignment.
    No local download required!
    """
    
    def __init__(
        self,
        tokenizer,
        bed_file,
        msa_path="zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip",
        max_len=512,
        num_views=2,
        mask_ratio=0.3,
        ortho_prob=0.5,
        species_pool=None,
        use_rc=True,
        debug=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.ortho_prob = ortho_prob
        self.use_rc = use_rc
        self.debug = debug
        
        # Load GenomeMSA with streaming path format
        print(f"Initializing GenomeMSA from remote zarr...")
        print(f"  Path: {msa_path}")
        print(f"  Note: First query will be slow (~10GB loaded into memory)")
        
        if not GENOMEMSA_AVAILABLE:
            raise ImportError("GenomeMSA not available. Install gpn package first.")
        
        try:
            self.genome_msa = GenomeMSA(msa_path)
            print("  ✓ GenomeMSA initialized in streaming mode")
        except Exception as e:
            print(f"  ✗ Failed to initialize GenomeMSA: {e}")
            raise
        
        # Load genomic coordinates from pre-built BED file
        print(f"Loading coordinates from {bed_file}...")
        self.coordinates = self._load_bed(bed_file)
        print(f"  ✓ Loaded {len(self.coordinates)} functional regions")
        
        # Species selection
        if species_pool is None:
            self.species_pool = list(range(11, 90))
        else:
            self.species_pool = species_pool
        
        print(f"  ✓ Species pool: {len(self.species_pool)} species")
        
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.rc_map = {
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
        }
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'ortho_success': 0,
            'ortho_failed': 0,
            'rc_used': 0,
        }
    
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
                
                # Get type/name if available (for ENCODE cCREs)
                region_type = parts[3] if len(parts) > 3 else 'unknown'
                strand = '+' if len(parts) < 6 else parts[5]
                
                # Split long regions
                step_size = max(self.max_len // 2, 100)
                for chunk_start in range(start, end, step_size):
                    chunk_end = min(chunk_start + self.max_len, end)
                    if chunk_end - chunk_start < 100:
                        continue
                    coords.append({
                        'chrom': chrom,
                        'start': chunk_start,
                        'end': chunk_end,
                        'strand': strand,
                        'type': region_type
                    })
        return coords
    
    def _array_to_seq(self, arr):
        """Convert GenomeMSA array to DNA string."""
        if isinstance(arr, np.ndarray):
            # Check what type of data we have
            if arr.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object strings
                # Array of bytes/strings like b'A', b'C', etc.
                return ''.join(x.decode() if isinstance(x, bytes) else str(x) for x in arr)
            elif arr.dtype.kind in ['i', 'u']:  # Integer tokens
                # Token mapping: 0=gap, 1=A, 2=C, 3=G, 4=T, 5=N
                token_map = {0: '-', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}
                return ''.join(token_map.get(int(t), 'N') for t in arr)
            else:
                # Unknown format, try string conversion
                return ''.join(str(x) for x in arr)
        elif isinstance(arr, str):
            return arr
        elif isinstance(arr, bytes):
            return arr.decode()
        else:
            return str(arr)

    
    def _get_orthologous_sequence(self, coord):
        """
        Stream orthologous sequence from remote zarr.
        Returns orthologous seq and human seq without downloading full dataset.
        """
        try:
            if self.debug:
                print(f"    Fetching MSA for chr{coord['chrom']}:{coord['start']}-{coord['end']} ({coord.get('type', 'unknown')})")
            
            # GenomeMSA.get_msa returns tokenized numpy arrays
            msa = self.genome_msa.get_msa(
                chrom=coord['chrom'],
                start=coord['start'],
                end=coord['end'],
                strand=coord['strand'],
                tokenize=False
            )
            
            if self.debug:
                print(f"    MSA retrieved: {len(msa)} species, length {len(msa[0]) if len(msa) > 0 else 0}")
            
            # Human is index 0
            if len(msa) == 0:
                return None, None
            
            human_seq_arr = msa[0]
            human_seq_str = self._array_to_seq(human_seq_arr)
            
            # Sample valid species with <50% gaps
            valid_species = []
            for sp_idx in self.species_pool:
                if sp_idx < len(msa):
                    sp_seq_arr = msa[sp_idx]
                    sp_seq_str = self._array_to_seq(sp_seq_arr)
                    gap_ratio = sp_seq_str.count('-') / len(sp_seq_str) if len(sp_seq_str) > 0 else 1.0
                    if gap_ratio < 0.5:
                        valid_species.append((sp_idx, gap_ratio))
            
            if not valid_species:
                if self.debug:
                    print(f"    No valid species found (all have >50% gaps)")
                return None, human_seq_str.replace('-', '')
            
            # Sort by gap ratio and pick from top candidates
            valid_species.sort(key=lambda x: x[1])
            species_idx = random.choice([s[0] for s in valid_species[:min(10, len(valid_species))]])
            ortho_seq_arr = msa[species_idx]
            ortho_seq_str = self._array_to_seq(ortho_seq_arr)
            
            if self.debug:
                print(f"    Selected species {species_idx} with {valid_species[0][1]*100:.1f}% gaps")
            
            # Remove gaps
            ortho_seq_clean = ortho_seq_str.replace('-', '')
            human_seq_clean = human_seq_str.replace('-', '')
            
            if self.debug:
                print(f"    Human seq: {len(human_seq_clean)} bp")
                print(f"    Ortho seq: {len(ortho_seq_clean)} bp")
                print(f"    ✓ Successfully retrieved ortholog")
            
            return ortho_seq_clean, human_seq_clean
            
        except Exception as e:
            if self.debug:
                import traceback
                print(f"    ✗ Failed to get MSA: {e}")
                traceback.print_exc()
            return None, None
    
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
            
            mask[start:start + block_size] = False
            num_masked += block_size
        
        return mask
    
    def _make_views_with_orthologs(self, coord):
        """Generate views with orthologous augmentation."""
        use_ortho = random.random() < self.ortho_prob
        
        ortho_seq, human_seq = None, None
        if use_ortho:
            ortho_seq, human_seq = self._get_orthologous_sequence(coord)
            
            if ortho_seq is not None and len(ortho_seq) > 50:
                self.stats['ortho_success'] += 1
            else:
                self.stats['ortho_failed'] += 1
        
        # Skip if MSA fetch completely failed
        if human_seq is None or len(human_seq) < 50:
            return None
        
        views = []
        used_ortho = False
        used_rc = False
        
        for v in range(self.num_views):
            if v == 0:
                src_text = human_seq
            elif v >= 1 and ortho_seq is not None and len(ortho_seq) > 50 and not used_ortho:
                src_text = ortho_seq
                used_ortho = True
            elif self.use_rc and random.random() < 0.5:
                src_text = self._reverse_complement(human_seq)
                used_rc = True
            else:
                src_text = human_seq
            
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
        
        if used_rc:
            self.stats['rc_used'] += 1
        
        self.stats['total_samples'] += 1
        
        metadata = {
            'coord': coord,
            'used_ortho': used_ortho,
            'used_rc': used_rc,
            'human_len': len(human_seq),
            'ortho_len': len(ortho_seq) if ortho_seq else 0,
        }
        
        return torch.stack(views), metadata
    
    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        
        my_coords_idx = list(range(worker_id, len(self.coordinates), num_workers))
        
        if len(my_coords_idx) == 0:
            return iter([])
        
        rng = random.Random(torch.initial_seed() % (2**32))
        
        for idx in itertools.cycle(my_coords_idx):
            if idx % 1000 == 0:
                rng.shuffle(my_coords_idx)
            
            coord = self.coordinates[idx]
            result = self._make_views_with_orthologs(coord)
            
            if result is not None:
                yield result


# Rest of the test functions remain the same...
def download_bed_files():
    """Download pre-built functional regions BED files."""
    print("\n" + "="*60)
    print("STEP 1: Downloading BED files")
    print("="*60)
    
    bed_dir = Path("./bed_files")
    bed_dir.mkdir(exist_ok=True)
    
    bed_files = {
        "promoters": "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.PLS.bed",
        "enhancers": "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.ELS.bed",
    }
    
    downloaded_files = []
    
    for name, url in bed_files.items():
        output_path = bed_dir / f"{name}.bed"
        
        if output_path.exists():
            print(f"  ✓ {name}.bed already exists")
            downloaded_files.append(output_path)
            continue
        
        print(f"  Downloading {name} from ENCODE SCREEN...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            print(f"    ✓ Downloaded to {output_path}")
            downloaded_files.append(output_path)
        except Exception as e:
            print(f"    ✗ Failed to download: {e}")
    
    combined_bed = bed_dir / "functional_regions_combined.bed"
    if combined_bed.exists():
        print(f"\n  ✓ Combined BED file already exists: {combined_bed}")
    else:
        print(f"\n  Combining BED files...")
        with open(combined_bed, 'w') as out_f:
            for bed_file in downloaded_files:
                with open(bed_file, 'r') as in_f:
                    out_f.write(in_f.read())
        print(f"    ✓ Created {combined_bed}")
    
    num_regions = sum(1 for line in open(combined_bed) 
                     if not line.startswith('#') and not line.startswith('track'))
    print(f"    Total regions: {num_regions:,}")
    
    return combined_bed


def test_bed_file(bed_file):
    """Test BED file loading and show statistics."""
    print("\n" + "="*60)
    print("STEP 2: Testing BED file")
    print("="*60)
    
    coords = []
    chrom_counts = defaultdict(int)
    
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
            
            coords.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'length': end - start
            })
            chrom_counts[chrom] += 1
    
    print(f"  ✓ Loaded {len(coords):,} regions")
    print(f"\n  Chromosome distribution (top 5):")
    for chrom, count in sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    chr{chrom}: {count:,} regions")
    
    lengths = [c['length'] for c in coords]
    print(f"\n  Region length statistics:")
    print(f"    Mean: {np.mean(lengths):.0f} bp")
    print(f"    Median: {np.median(lengths):.0f} bp")
    
    return coords


def test_tokenizer():
    """Test tokenizer loading."""
    print("\n" + "="*60)
    print("STEP 3: Testing tokenizer")
    print("="*60)
    
    print("  Loading DNABERT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", 
        trust_remote_code=True,
        cache_dir="./"
    )
    print(f"    ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    return tokenizer


def test_dataset_creation(bed_file, tokenizer):
    """Test dataset creation."""
    print("\n" + "="*60)
    print("STEP 4: Testing dataset creation")
    print("="*60)
    
    dataset = OrthologousAugmentedDataset(
        tokenizer=tokenizer,
        bed_file=str(bed_file),
        msa_path="zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip",
        max_len=512,
        num_views=2,
        mask_ratio=0.3,
        ortho_prob=0.5,
        species_pool=list(range(11, 90)),
        use_rc=True,
        debug=True,
    )
    print(f"    ✓ Dataset created")
    return dataset


def test_dataloader(dataset, num_samples=3):
    """Test dataloader iteration."""
    print("\n" + "="*60)
    print("STEP 5: Testing dataloader (this will take 1-2 min for first query)")
    print("="*60)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    for i, batch_data in enumerate(loader):
        if i >= num_samples:
            break
        
        batch_views, batch_metadata = batch_data
        
        print(f"\n  --- Batch {i+1} ---")
        print(f"  Shape: {batch_views.shape}")
        
        for b in range(batch_views.shape[0]):
            metadata = batch_metadata[b] if isinstance(batch_metadata, (list, tuple)) else None
            if metadata:
                print(f"    Sample {b+1}: chr{metadata['coord']['chrom']}:{metadata['coord']['start']}")
                print(f"      Ortho used: {metadata['used_ortho']}, RC used: {metadata['used_rc']}")
    
    print(f"\n  Dataset statistics:")
    print(f"    Total: {dataset.stats['total_samples']}")
    print(f"    Ortho success: {dataset.stats['ortho_success']}")
    print(f"    Ortho failed: {dataset.stats['ortho_failed']}")
    
    if (dataset.stats['ortho_success'] + dataset.stats['ortho_failed']) > 0:
        rate = dataset.stats['ortho_success'] / (dataset.stats['ortho_success'] + dataset.stats['ortho_failed']) * 100
        print(f"    Success rate: {rate:.1f}%")


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("ORTHOLOGOUS DATASET TEST SUITE")
    print("="*60)
    
    try:
        bed_file = download_bed_files()
        coords = test_bed_file(bed_file)
        tokenizer = test_tokenizer()
        dataset = test_dataset_creation(bed_file, tokenizer)
        test_dataloader(dataset, num_samples=3)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour dataset is ready!")
        print(f"BED file: {bed_file}")
        print(f"Regions: {len(coords):,}")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
