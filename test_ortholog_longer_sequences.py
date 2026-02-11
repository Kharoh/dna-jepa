# test_longer_sequences.py
"""
Test if GenomeMSA can return sequences longer than 90bp.
"""

import numpy as np
from gpn.data import GenomeMSA

print("Testing GenomeMSA sequence length capabilities...")
print("="*60)

# Initialize GenomeMSA
msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"
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
        print(f"    Number of species: {len(msa)}")
        print(f"    Sequence length: {len(msa[0]) if len(msa) > 0 else 0} bp")
        print(f"    Requested length: {region['end'] - region['start']} bp")
        
        # Check if length matches request
        if len(msa) > 0:
            actual_len = len(msa[0])
            expected_len = region['end'] - region['start']
            if actual_len == expected_len:
                print(f"    ✓ Length matches! GenomeMSA returned full sequence")
            else:
                print(f"    ⚠ Length mismatch: got {actual_len}, expected {expected_len}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*60)
print("CONCLUSION:")
print("If all tests show matching lengths, GenomeMSA can handle")
print("arbitrary sequence lengths and the 90bp you saw was just")
print("due to the specific regions in your BED file!")
