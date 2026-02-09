import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os

# ==========================================
# 1. YOUR EXACT DATASET CLASS (COPY-PASTED)
# ==========================================
class DNAJEPADataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_len=512, num_views=2, mask_ratio=0.3):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_views = num_views
        self.mask_ratio = mask_ratio

        print(f"Loading dataset from {txt_path} using Memory Mapping...")
        # Use load_dataset to memory-map the file instead of reading into RAM
        # keep_in_memory=False ensures it stays on disk
        # We add trust_remote_code=True if needed, though 'text' dataset usually doesn't need it
        dataset = load_dataset("text", data_files={"train": txt_path}, split="train", keep_in_memory=False, cache_dir="./")
        self.lines = dataset  
        print(f"Loaded {len(self.lines)} sequences.")

    def __len__(self):
        return len(self.lines)

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

    def __getitem__(self, idx):
        text = self.lines[idx]['text']

        # Generate V different views (same tokenization, different masks)
        views = []
        for _ in range(self.num_views):
            ids = self._encode_no_dropout(text)
            ids = ids[:self.max_len]
            padding = [self.tokenizer.pad_token_id] * (self.max_len - len(ids))
            view_tokens = torch.tensor(ids + padding, dtype=torch.long)

            seq_len = min(len(ids), self.max_len)
            mask_content = self._get_block_mask(seq_len, mask_ratio=self.mask_ratio)
            full_mask = torch.ones(self.max_len, dtype=torch.bool)
            full_mask[:seq_len] = mask_content

            masked_view = view_tokens.clone()
            masked_view[~full_mask] = self.tokenizer.pad_token_id

            views.append(masked_view)

        return torch.stack(views)

# ==========================================
# 2. THE DIAGNOSTIC SCRIPT
# ==========================================
def main():
    # --- CONFIG ---
    TRAIN_FILE = "train.txt"  # <--- Make sure this matches your file name
    MAX_LEN = 512
    MASK_RATIO = 0.3
    VIEWS = 2
    BATCH_SIZE = 4
    
    # Check file existence
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Error: Could not find '{TRAIN_FILE}'. Please verify the path.")
        return

    print("\n>>> 1. Initializing Tokenizer (DNABERT-2)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        print(f"   Vocab Size: {tokenizer.vocab_size}")
        print(f"   Pad Token ID: {tokenizer.pad_token_id}")
        if tokenizer.pad_token_id is None:
            print("   ⚠️  WARNING: tokenizer.pad_token_id is None. Setting to 0 for this test.")
            tokenizer.pad_token_id = 0
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    print("\n>>> 2. Building Dataset...")
    try:
        ds = DNAJEPADataset(TRAIN_FILE, tokenizer, max_len=MAX_LEN, num_views=VIEWS, mask_ratio=MASK_RATIO)
    except Exception as e:
        print(f"❌ Failed to initialize Dataset: {e}")
        return

    print("\n>>> 3. Inspecting Random Samples...")
    
    # Create a small loader
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get one batch
    try:
        batch = next(iter(loader)) # Shape: (B, V, L)
    except Exception as e:
        print(f"❌ Failed to fetch batch from DataLoader: {e}")
        return

    print(f"   Batch Shape: {batch.shape} (Batch, Views, SeqLen)")
    
    # Analyze the first sample in the batch
    # We look at View 0 and View 1 to verify masking differences
    view1_ids = batch[0, 0].tolist()
    view2_ids = batch[0, 1].tolist()
    
    def decode_and_stats(ids, name):
        # Remove padding for clear view
        valid_ids = [t for t in ids if t != tokenizer.pad_token_id]
        text = tokenizer.decode(valid_ids)
        
        print(f"\n   [{name}]")
        print(f"   - Total Length (incl pad): {len(ids)}")
        print(f"   - Active Tokens (unmasked): {len(valid_ids)}")
        print(f"   - Padding Tokens: {len(ids) - len(valid_ids)}")
        print(f"   - First 100 chars decoded: {text[:100]}")
        
        # CHECKS
        if len(valid_ids) == 0:
            print("     🚨 ALARM: View is completely empty (all padding)!")
        elif len(valid_ids) < 10:
             print("     ⚠️  WARNING: Very short sequence.")
        
        if ">" in text:
             print("     🚨 ALARM: Found FASTA header ('>') in data. Clean your txt file!")
        
        if "[UNK]" in text:
             print("     ⚠️  WARNING: Found [UNK] tokens. Check for 'N' or non-ACGT chars.")
             
        return set(valid_ids), text

    print("\n   --- Comparison of Views for Sample 0 ---")
    v1_tokens, v1_text = decode_and_stats(view1_ids, "View 1")
    v2_tokens, v2_text = decode_and_stats(view2_ids, "View 2")
    
    # Verify diversity
    if view1_ids == view2_ids:
        print("\n   🚨 CRITICAL ALARM: View 1 and View 2 are IDENTICAL.")
        print("      The masking logic is not generating diversity.")
    else:
        print("\n   ✅ SUCCESS: View 1 and View 2 are different.")
        
    # Verify overlaps (JEPAs need some overlap, but not 100%)
    intersection = v1_tokens.intersection(v2_tokens)
    overlap_ratio = len(intersection) / max(len(v1_tokens), 1)
    print(f"   - Token Overlap Ratio: {overlap_ratio:.2f} (Should be < 1.0 but > 0.0)")

    print("\n>>> 4. Full Dataset Statistics (Quick Scan)")
    # Optional: check lengths of first 100 items to see distribution
    lengths = []
    scan_limit = min(100, len(ds))
    print(f"   Scanning first {scan_limit} items for length distribution...")
    
    for i in range(scan_limit):
        item = ds[i] # (V, L)
        # Just check view 0 length
        valid = (item[0] != tokenizer.pad_token_id).sum().item()
        lengths.append(valid)
        
    avg_len = sum(lengths) / len(lengths)
    print(f"   - Average Active Tokens: {avg_len:.1f} / {MAX_LEN}")
    print(f"   - Min Length: {min(lengths)}")
    print(f"   - Max Length: {max(lengths)}")
    
    if avg_len < 50:
        print("   ⚠️  WARNING: Your data seems very short on average. Is this intended?")
    
    print("\n>>> Done.")

if __name__ == "__main__":
    main()
