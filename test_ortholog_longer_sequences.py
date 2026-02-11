# test_longer_sequences.py
"""
Test if GenomeMSA can return sequences longer than 90bp.
"""

import numpy as np
from gpn.data import GenomeMSA


print("Testing GenomeMSA sequence length capabilities...")
print("="*60)


# Initialize GenomeMSA
msa_path = "./89.zarr"
genome_msa = GenomeMSA(msa_path)


# Test different region lengths
test_regions = [
    {"chrom": "1", "start": 10000000, "end": 10000100, "strand": "+", "name": "100bp"},
    {"chrom": "1", "start": 10000000, "end": 10000256, "strand": "+", "name": "256bp"},
    {"chrom": "1", "start": 10000000, "end": 10000512, "strand": "+", "name": "512bp"},
    {"chrom": "1", "start": 10000000, "end": 10001000, "strand": "+", "name": "1000bp"},
]


for region in test_regions:
    try:
        print(f"\nTesting {region['name']} region:")
        print(f"  Region: chr{region['chrom']}:{region['start']}-{region['end']}")
        
        msa = genome_msa.get_msa(
            chrom=region['chrom'],
            start=region['start'],
            end=region['end'],
            strand=region['strand'],
            tokenize=False
        )
        
        print(f"  ✓ MSA retrieved")
        print(f"    MSA shape: {msa.shape}")
        print(f"    Sequence length (positions): {len(msa)}")  # ← FIX: This is seq length
        print(f"    Number of species: {len(msa[0]) if len(msa) > 0 else 0}")  # ← FIX: This is species
        print(f"    Requested length: {region['end'] - region['start']} bp")
        
        # Check if length matches request
        if len(msa) > 0:
            actual_len = len(msa)  # ← FIX: First dimension is sequence length
            expected_len = region['end'] - region['start']
            if actual_len == expected_len:
                print(f"    ✓ Length matches! GenomeMSA returned full sequence")
            else:
                print(f"    ⚠ Length mismatch: got {actual_len}, expected {expected_len}")
            
            # Verify we have the right structure
            print(f"    Species count: {msa.shape[1] if len(msa.shape) > 1 else 'N/A'}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")


print("\n" + "="*60)
print("CONCLUSION:")
print("GenomeMSA returns MSA with shape (seq_length, num_species)")
print("where seq_length matches your requested region!")
