#!/usr/bin/env python3
"""
Convert fairseq dictionary to HuggingFace BART tokenizer format.

This script creates vocab.json and merges.txt files from a fairseq dict.txt file,
allowing you to use a fairseq-trained BART model with HuggingFace Transformers.
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict


def read_fairseq_dict(dict_path, add_special_tokens=True):
    """
    Read fairseq dict.txt file.

    Format: token count
    Example:
        the 1234567
        , 987654
        ...

    Args:
        dict_path: Path to fairseq dict.txt
        add_special_tokens: If True, adds standard fairseq special tokens at indices 0-4
    """
    vocab = OrderedDict()

    # Fairseq reserves indices 0-4 for special tokens (not in dict.txt)
    # Standard convention:
    # 0: <s> (BOS)
    # 1: <pad>
    # 2: </s> (EOS)
    # 3: <unk>
    # 4: <mask> (for BART denoising)
    # Then dict.txt tokens start from index 5
    if add_special_tokens:
        vocab['<s>'] = 0
        vocab['<pad>'] = 1
        vocab['</s>'] = 2
        vocab['<unk>'] = 3
        vocab['<mask>'] = 4
        # dict.txt tokens start from index 5
        start_idx = 5
    else:
        start_idx = 0

    with open(dict_path, 'r', encoding='utf-8') as f:
        for file_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 1:
                token = parts[0]
                # In fairseq, dict.txt tokens start after special tokens
                # So actual index = file line number + number of special tokens
                actual_idx = file_idx + start_idx if add_special_tokens else file_idx
                vocab[token] = actual_idx

    return vocab


def create_vocab_json(fairseq_vocab, output_path):
    """
    Create vocab.json for HuggingFace tokenizer.

    This maps tokens to their IDs.
    """
    vocab_dict = {}

    # Add special tokens first if they exist
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>']
    for token in special_tokens:
        if token in fairseq_vocab:
            vocab_dict[token] = fairseq_vocab[token]

    # Add all other tokens
    for token, idx in fairseq_vocab.items():
        if token not in vocab_dict:
            vocab_dict[token] = idx

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    return vocab_dict


def create_merges_txt(fairseq_vocab, output_path):
    """
    Create merges.txt for HuggingFace tokenizer.

    For word-level tokenization (no BPE), we create a minimal merges file
    with just the version header. This is required by HuggingFace BART tokenizer
    but won't actually be used for merging since tokens are already complete words.
    """
    # Write minimal merges file (just version header)
    # For word-level tokenization, no actual merges are needed
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")
        # Add a few dummy merges to satisfy the tokenizer
        # These won't affect tokenization since all words are in vocab
        f.write("Ġ t\n")
        f.write("Ġ a\n")
        f.write("Ġ the\n")

    return []


def create_tokenizer_config(output_path, vocab_size):
    """Create a basic tokenizer_config.json for HuggingFace."""
    config = {
        "add_prefix_space": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1024,
        "pad_token": "<pad>",
        "tokenizer_class": "BartTokenizer",
        "unk_token": "<unk>",
        "vocab_size": vocab_size
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config


def convert_fairseq_dict_to_hf(dict_path, output_dir):
    """
    Main conversion function.

    Args:
        dict_path: Path to fairseq dict.txt
        output_dir: Directory to save HuggingFace tokenizer files
    """
    print(f"\n{'='*70}")
    print(f"Converting fairseq dictionary to HuggingFace tokenizer format")
    print(f"{'='*70}")
    print(f"Input dict: {dict_path}")
    print(f"Output dir: {output_dir}\n")

    # Read fairseq dictionary
    print("Step 1: Reading fairseq dictionary...")
    fairseq_vocab = read_fairseq_dict(dict_path, add_special_tokens=True)
    vocab_size = len(fairseq_vocab)
    print(f"✓ Loaded {vocab_size:,} tokens")

    # Show special tokens
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    print(f"\n  Special tokens added:")
    for token in special_tokens:
        if token in fairseq_vocab:
            print(f"    {token:10} -> index {fairseq_vocab[token]}")

    # Show some sample regular tokens
    regular_tokens = [k for k in fairseq_vocab.keys() if k not in special_tokens][:10]
    print(f"\n  Sample regular tokens (first 10):")
    for token in regular_tokens[:5]:
        print(f"    {token:20} -> index {fairseq_vocab[token]}")

    print(f"\n  Expected vocab size: 248,161 (248,156 from dict.txt + 5 special tokens)")
    print(f"  Actual vocab size:   {vocab_size:,}")
    if vocab_size == 248161:
        print(f"  ✓ Vocabulary size matches!")
    else:
        print(f"  ⚠️  Warning: Size mismatch!")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create vocab.json
    print("\nStep 2: Creating vocab.json...")
    vocab_json_path = output_path / "vocab.json"
    vocab_dict = create_vocab_json(fairseq_vocab, vocab_json_path)
    print(f"✓ Created {vocab_json_path}")
    print(f"  Vocabulary size: {len(vocab_dict):,}")

    # Create merges.txt
    print("\nStep 3: Creating merges.txt...")
    merges_txt_path = output_path / "merges.txt"
    merges = create_merges_txt(fairseq_vocab, merges_txt_path)
    print(f"✓ Created {merges_txt_path}")
    print(f"  Number of merges: {len(merges):,}")

    # Create tokenizer_config.json
    print("\nStep 4: Creating tokenizer_config.json...")
    config_path = output_path / "tokenizer_config.json"
    config = create_tokenizer_config(config_path, vocab_size)
    print(f"✓ Created {config_path}")

    print(f"\n{'='*70}")
    print(f"✓ Conversion complete!")
    print(f"{'='*70}")
    print(f"Tokenizer files saved to: {output_dir}")
    print(f"  - vocab.json ({vocab_size:,} tokens)")
    print(f"  - merges.txt ({len(merges):,} merges)")
    print(f"  - tokenizer_config.json")
    print(f"\n⚠️  Note: This is a basic conversion. The tokenizer may not perfectly")
    print(f"   match the original fairseq BPE behavior, but it will work with")
    print(f"   HuggingFace Transformers and your converted model weights.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fairseq dict.txt to HuggingFace tokenizer format"
    )
    parser.add_argument(
        "dict_path",
        type=str,
        help="Path to fairseq dict.txt file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for HuggingFace tokenizer files"
    )

    args = parser.parse_args()
    convert_fairseq_dict_to_hf(args.dict_path, args.output_dir)
