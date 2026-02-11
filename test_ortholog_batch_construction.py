# test_ortholog_batch_construction.py
"""
Test script for ortholog batch construction with detailed timing.
Diagnoses bottlenecks in the ortholog augmentation pipeline.

Usage:
    python test_ortholog_batch_construction.py
"""

import os
import sys
import time
import torch
import random
import numpy as np
from transformers import AutoTokenizer

# Try to import GenomeMSA
try:
    from gpn.data import GenomeMSA
    GENOMEMSA_AVAILABLE = True
    print("✓ GenomeMSA available")
except ImportError:
    GENOMEMSA_AVAILABLE = False
    print("❌ GenomeMSA not available")
    sys.exit(1)


class OrthologHelper:
    """Helper class with detailed debugging."""
    
    def __init__(
        self,
        bed_file,
        msa_path="zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip",
        window_size=90,
        min_alignment_quality=0.25, # FIXME: originally 0.4
        debug=True,
    ):
        self.window_size = window_size
        self.min_alignment_quality = min_alignment_quality
        self.debug = debug
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'msa_fetch_success': 0,
            'msa_fetch_failed': 0,
            'no_valid_species': 0,
            'sequences_too_short': 0,
            'success': 0,
        }
        
        print(f"Loading GenomeMSA from {msa_path}...")
        start = time.time()
        self.genome_msa = GenomeMSA(msa_path)
        print(f"  ✓ GenomeMSA loaded in {time.time() - start:.2f}s")
        
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
    
    def get_ortholog_sequence(self, max_len=512, species_preference='mixed'):
        """Get ortholog with detailed debugging."""
        max_attempts = 20
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"ORTHOLOG FETCH START (max_len={max_len}, pref={species_preference})")
            print(f"{'='*60}")
        
        for attempt in range(max_attempts):
            self.stats['total_attempts'] += 1
            attempt_start = time.time()
            
            if self.debug:
                print(f"\n--- Attempt {attempt+1}/{max_attempts} ---")
            
            # STEP 1: Pick random region
            coord = random.choice(self.coordinates)
            region_len = coord['end'] - coord['start']
            
            if self.debug:
                print(f"  [1] Selected region: chr{coord['chrom']}:{coord['start']}-{coord['end']} ({region_len}bp)")
            
            # STEP 2: Crop to max_len
            if region_len > max_len:
                offset = random.randint(0, region_len - max_len)
                query_start = coord['start'] + offset
                query_end = query_start + max_len
            else:
                query_start = coord['start']
                query_end = coord['end']
            
            actual_query_len = query_end - query_start
            if self.debug:
                print(f"  [2] Query region: chr{coord['chrom']}:{query_start}-{query_end} ({actual_query_len}bp)")
            
            # STEP 3: Query MSA in windows
            try:
                all_human_seqs = []
                all_ortho_seqs = []
                selected_species = None
                
                num_windows = (query_end - query_start + self.window_size - 1) // self.window_size
                if self.debug:
                    print(f"  [3] Querying {num_windows} windows of {self.window_size}bp...")
                
                window_count = 0
                for chunk_start in range(query_start, query_end, self.window_size):
                    chunk_end = min(chunk_start + self.window_size, query_end)
                    window_count += 1
                    
                    if self.debug and window_count == 1:
                        msa_start = time.time()
                    
                    try:
                        msa = self.genome_msa.get_msa(
                            chrom=coord['chrom'],
                            start=chunk_start,
                            end=chunk_end,
                            strand=coord['strand'],
                            tokenize=False
                        )
                        
                        if self.debug and window_count == 1:
                            msa_elapsed = time.time() - msa_start
                            print(f"      Window 1 MSA query: {msa_elapsed:.2f}s")
                            print(f"      MSA shape: {len(msa)} species x {len(msa[0]) if len(msa) > 0 else 0}bp")
                        
                        if len(msa) == 0:
                            if self.debug:
                                print(f"      ⚠ Window {window_count}: Empty MSA")
                            continue
                        
                        self.stats['msa_fetch_success'] += 1
                        
                        # Get human sequence
                        human_seq_str = self._array_to_seq(msa[0]).replace('-', '')
                        all_human_seqs.append(human_seq_str)
                        
                        if self.debug and window_count == 1:
                            print(f"      Human seq (window 1): {len(human_seq_str)}bp after gap removal")
                        
                        # STEP 4: Select species on first chunk
                        if selected_species is None:
                            species_select_start = time.time()
                            
                            if species_preference == 'close':
                                species_pool = self.close_species
                            elif species_preference == 'distant':
                                species_pool = self.distant_species
                            else:
                                species_pool = self.all_species
                            
                            if self.debug:
                                print(f"  [4] Filtering {len(species_pool)} candidate species...")
                            
                            # Find valid species
                            valid_species = []
                            for sp_idx in species_pool:
                                if sp_idx < len(msa):
                                    sp_seq_str = self._array_to_seq(msa[sp_idx])
                                    alignment_quality = 1.0 - (sp_seq_str.count('-') / len(sp_seq_str))
                                    if alignment_quality >= self.min_alignment_quality:
                                        valid_species.append((sp_idx, alignment_quality))
                            
                            species_elapsed = time.time() - species_select_start
                            
                            if self.debug:
                                print(f"      Found {len(valid_species)}/{len(species_pool)} valid species ({species_elapsed:.3f}s)")
                                if len(valid_species) > 0:
                                    print(f"      Best alignment quality: {max(v[1] for v in valid_species):.2%}")
                                    print(f"      Worst alignment quality: {min(v[1] for v in valid_species):.2%}")
                            
                            if not valid_species:
                                self.stats['no_valid_species'] += 1
                                if self.debug:
                                    print(f"      ✗ No valid species (all <{self.min_alignment_quality:.0%} alignment)")
                                break
                            
                            # Pick best quality species
                            valid_species.sort(key=lambda x: x[1], reverse=True)
                            selected_species = random.choice([s[0] for s in valid_species[:min(5, len(valid_species))]])
                            
                            if self.debug:
                                selected_quality = [v[1] for v in valid_species if v[0] == selected_species][0]
                                print(f"      ✓ Selected species {selected_species} (quality: {selected_quality:.2%})")
                        
                        # Get ortholog sequence
                        if selected_species < len(msa):
                            ortho_seq_str = self._array_to_seq(msa[selected_species]).replace('-', '')
                            all_ortho_seqs.append(ortho_seq_str)
                    
                    except Exception as e:
                        self.stats['msa_fetch_failed'] += 1
                        if self.debug:
                            print(f"      ✗ Window {window_count} MSA error: {e}")
                        continue
                
                # STEP 5: Concatenate and validate
                full_human = ''.join(all_human_seqs)
                full_ortho = ''.join(all_ortho_seqs)
                
                if self.debug:
                    print(f"  [5] Concatenated sequences:")
                    print(f"      Human: {len(full_human)}bp")
                    print(f"      Ortho: {len(full_ortho)}bp")
                
                # Check length
                if len(full_human) >= 50 and len(full_ortho) >= 50:
                    attempt_elapsed = time.time() - attempt_start
                    self.stats['success'] += 1
                    
                    if self.debug:
                        print(f"  [6] ✓ SUCCESS in {attempt_elapsed:.2f}s")
                        print(f"{'='*60}\n")
                    
                    metadata = {
                        'coord': coord,
                        'species_idx': selected_species,
                        'human_len': len(full_human),
                        'ortho_len': len(full_ortho),
                        'attempt': attempt + 1,
                    }
                    return full_human, full_ortho, metadata
                else:
                    self.stats['sequences_too_short'] += 1
                    if self.debug:
                        print(f"      ✗ Sequences too short (need ≥50bp)")
                
            except Exception as e:
                if self.debug:
                    print(f"  ✗ Attempt failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Failed after all attempts
        if self.debug:
            print(f"\n✗ FAILED after {max_attempts} attempts")
            self.print_stats()
        
        return None
    
    def print_stats(self):
        """Print cumulative statistics."""
        print(f"\n{'='*60}")
        print("ORTHOLOG HELPER STATISTICS")
        print(f"{'='*60}")
        print(f"Total attempts: {self.stats['total_attempts']}")
        print(f"MSA fetch success: {self.stats['msa_fetch_success']}")
        print(f"MSA fetch failed: {self.stats['msa_fetch_failed']}")
        print(f"No valid species: {self.stats['no_valid_species']}")
        print(f"Sequences too short: {self.stats['sequences_too_short']}")
        print(f"Total successes: {self.stats['success']}")
        if self.stats['success'] > 0:
            avg_attempts = self.stats['total_attempts'] / self.stats['success']
            print(f"Average attempts per success: {avg_attempts:.1f}")
        print(f"{'='*60}\n")


class OrthologCollator:
    """Collator for testing batch construction."""
    
    def __init__(self, tokenizer, ortholog_helper, max_len=512, num_views=2, mask_ratio=0.3):
        self.tokenizer = tokenizer
        self.ortholog_helper = ortholog_helper
        self.max_len = max_len
        self.num_views = num_views
        self.mask_ratio = mask_ratio
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        self.rc_map = {
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
        }
    
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
    
    def _make_views_ortholog(self):
        """Ortholog views: different species for each view."""
        views = []
        
        for v in range(self.num_views):
            species_pref = 'close' if v % 2 == 0 else 'distant'
            
            result = self.ortholog_helper.get_ortholog_sequence(
                max_len=self.max_len,
                species_preference=species_pref
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
    
    def create_ortholog_batch(self, batch_size):
        """Create a full ortholog batch."""
        print(f"\n{'='*60}")
        print(f"CREATING ORTHOLOG BATCH (size={batch_size})")
        print(f"{'='*60}")
        
        batch_start = time.time()
        batch_views = []
        
        for idx in range(batch_size):
            sample_start = time.time()
            print(f"\n--- Sample {idx+1}/{batch_size} ---")
            
            views = self._make_views_ortholog()
            
            sample_elapsed = time.time() - sample_start
            
            if views is None:
                print(f"  ✗ Sample {idx+1} FAILED after {sample_elapsed:.2f}s")
                return None
            else:
                print(f"  ✓ Sample {idx+1} SUCCESS in {sample_elapsed:.2f}s")
            
            batch_views.append(views)
        
        batch_elapsed = time.time() - batch_start
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {batch_elapsed:.2f}s")
        print(f"Average per sample: {batch_elapsed/batch_size:.2f}s")
        print(f"Samples per second: {batch_size/batch_elapsed:.2f}")
        print(f"{'='*60}\n")
        
        return torch.stack(batch_views)


def test_single_fetch(ortholog_helper):
    """Test a single ortholog fetch."""
    print("\n" + "="*60)
    print("TEST 1: Single Ortholog Fetch")
    print("="*60)
    
    result = ortholog_helper.get_ortholog_sequence(max_len=512, species_preference='mixed')
    
    if result:
        human_seq, ortho_seq, metadata = result
        print("\n✓ SUCCESS")
        print(f"  Human: {len(human_seq)}bp")
        print(f"  Ortho: {len(ortho_seq)}bp (species {metadata['species_idx']})")
        print(f"  Attempts: {metadata['attempt']}")
    else:
        print("\n✗ FAILED")
    
    return result is not None


def test_multiple_fetches(ortholog_helper, num_fetches=5):
    """Test multiple ortholog fetches."""
    print("\n" + "="*60)
    print(f"TEST 2: Multiple Ortholog Fetches (n={num_fetches})")
    print("="*60)
    
    # Reset stats
    ortholog_helper.stats = {
        'total_attempts': 0,
        'msa_fetch_success': 0,
        'msa_fetch_failed': 0,
        'no_valid_species': 0,
        'sequences_too_short': 0,
        'success': 0,
    }
    
    # Disable detailed debug for this test
    ortholog_helper.debug = False
    
    start_time = time.time()
    successes = 0
    
    for i in range(num_fetches):
        print(f"\nFetch {i+1}/{num_fetches}...", end=' ')
        result = ortholog_helper.get_ortholog_sequence(max_len=512, species_preference='mixed')
        if result:
            successes += 1
            print("✓")
        else:
            print("✗")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Results: {successes}/{num_fetches} successful")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per fetch: {elapsed/num_fetches:.2f}s")
    print(f"{'='*60}")
    
    ortholog_helper.print_stats()
    
    return successes == num_fetches


def test_batch_construction(ortholog_helper, tokenizer, batch_size=4):
    """Test full batch construction."""
    print("\n" + "="*60)
    print(f"TEST 3: Batch Construction (batch_size={batch_size})")
    print("="*60)
    
    # Re-enable debug for batch test
    ortholog_helper.debug = True
    
    collator = OrthologCollator(
        tokenizer=tokenizer,
        ortholog_helper=ortholog_helper,
        max_len=512,
        num_views=2,
        mask_ratio=0.3
    )
    
    batch = collator.create_ortholog_batch(batch_size)
    
    if batch is not None:
        print(f"\n✓ Batch created successfully")
        print(f"  Shape: {batch.shape}")
        return True
    else:
        print(f"\n✗ Batch creation failed")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ORTHOLOG BATCH CONSTRUCTION TEST SUITE")
    print("="*60)
    
    # Configuration
    bed_file = "./bed_files/functional_regions_combined.bed"
    msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"
    
    # Check BED file exists
    if not os.path.exists(bed_file):
        print(f"\n✗ BED file not found: {bed_file}")
        print("Please download it first:")
        print("  wget https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.PLS.bed")
        print("  wget https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.ELS.bed")
        print("  mkdir -p bed_files")
        print("  cat GRCh38-cCREs.PLS.bed GRCh38-cCREs.ELS.bed > bed_files/functional_regions_combined.bed")
        sys.exit(1)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        cache_dir="./"
    )
    print(f"  ✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # Initialize ortholog helper
    print("\nInitializing OrthologHelper...")
    try:
        ortholog_helper = OrthologHelper(
            bed_file=bed_file,
            msa_path=msa_path,
            min_alignment_quality=0.4,
            debug=True
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize OrthologHelper: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run tests
    results = []
    
    # Test 1: Single fetch
    results.append(("Single Fetch", test_single_fetch(ortholog_helper)))
    
    # Test 2: Multiple fetches
    results.append(("Multiple Fetches", test_multiple_fetches(ortholog_helper, num_fetches=5)))
    
    # Test 3: Batch construction
    results.append(("Batch Construction", test_batch_construction(ortholog_helper, tokenizer, batch_size=4)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYour ortholog batch construction is working!")
        print("Expected performance:")
        ortholog_helper.print_stats()
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nCheck the logs above to diagnose the issue.")


if __name__ == "__main__":
    main()
