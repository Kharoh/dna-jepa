# https://www.perplexity.ai/search/import-os-import-torch-import-U437j.PuRROuYPiu74Vq9g#5

#!/usr/bin/env python
"""
Train a DNA BPE tokenizer with runtime dropout and export it
as a PreTrainedTokenizerFast-compatible tokenizer.

Usage:
    python make_dna_bpe_tokenizer.py --input train.txt --out_dir dna_bpe_tokenizer_dropout
"""

import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast


def train_dna_bpe_tokenizer(
    input_path: str,
    vocab_size: int = 32000,
    dropout: float = 0.1,
):
    # 1) Initialize an empty BPE model with dropout [web:15][web:9]
    bpe_model = BPE(unk_token="[UNK]", dropout=dropout)
    tokenizer = Tokenizer(bpe_model)

    # DNA is already “tokenless” text; we just split on whitespace/newlines
    tokenizer.pre_tokenizer = Whitespace()

    # 2) Trainer with standard special tokens
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    # 3) Train from file of sequences (one per line)
    tokenizer.train(files=[input_path], trainer=trainer)

    # 4) Add simple BERT-style post-processing (CLS/SEP) [web:43]
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    return tokenizer


def save_as_fast_tokenizer(tokenizer: Tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Save the raw tokenizers.Tokenizer (tokenizer.json)
    tokenizer_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    # Wrap as PreTrainedTokenizerFast so you can use AutoTokenizer.from_pretrained() [web:40][web:43]
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )

    fast_tok.save_pretrained(out_dir)
    print(f"Saved tokenizer to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to train.txt (one DNA seq per line)")
    parser.add_argument("--out_dir", type=str, default="dna_bpe_tokenizer_dropout", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Training BPE tokenizer on {args.input} with vocab={args.vocab_size}, dropout={args.dropout}")
    tok = train_dna_bpe_tokenizer(args.input, vocab_size=args.vocab_size, dropout=args.dropout)
    save_as_fast_tokenizer(tok, args.out_dir)


if __name__ == "__main__":
    main()
