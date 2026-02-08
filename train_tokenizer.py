#!/usr/bin/env python
"""
Train a DNA BPE tokenizer with dropout and export it as a
PreTrainedTokenizerFast-compatible tokenizer.

Usage:
    python make_dna_bpe_tokenizer.py \
        --input train.txt \
        --out_dir dna_bpe_tokenizer_dropout \
        --vocab_size 32000 \
        --dropout 0.1
"""

import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import PreTrainedTokenizerFast


def train_dna_bpe_tokenizer(
    input_path: str,
    vocab_size: int = 32000,
    dropout: float = 0.1,
) -> Tokenizer:
    """
    Train a BPE tokenizer on DNA sequences with BPE dropout enabled.
    Sequences are read from `input_path`, one per line.
    """
    # 1) Initialize BPE model with dropout (stochastic merges) [web:15][web:9]
    bpe_model = BPE(unk_token="[UNK]", dropout=dropout)
    tokenizer = Tokenizer(bpe_model)

    # DNA lines are treated as plain strings; split on whitespace/newlines
    tokenizer.pre_tokenizer = Whitespace()

    # 2) Train BPE vocab
    special_tokens = ["[UNK]", "[PAD]", "[MASK]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    print(f"Training BPE tokenizer on {input_path}")
    tokenizer.train(files=[input_path], trainer=trainer)

    # No post-processor: we don't auto-insert CLS/SEP.
    # Model will handle its own CLS embedding (as in your DNAEncoder).

    return tokenizer


def save_as_fast_tokenizer(tokenizer: Tokenizer, out_dir: str):
    """
    Save the tokenizers.Tokenizer and wrap it as PreTrainedTokenizerFast
    so it can be loaded with AutoTokenizer.from_pretrained(out_dir).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save raw tokenizer JSON
    tokenizer_json_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    # Wrap in a fast tokenizer for transformers [web:43][web:40]
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )

    fast_tok.save_pretrained(out_dir)
    print(f"Saved tokenizer to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to training text file (one DNA sequence per line).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dna_bpe_tokenizer_dropout",
        help="Output directory for the tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="BPE dropout probability (0.0 = deterministic BPE).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(
        f"Training DNA BPE tokenizer with "
        f"vocab_size={args.vocab_size}, dropout={args.dropout}"
    )
    tokenizer = train_dna_bpe_tokenizer(
        input_path=args.input,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
    )

    save_as_fast_tokenizer(tokenizer, args.out_dir)


if __name__ == "__main__":
    main()
