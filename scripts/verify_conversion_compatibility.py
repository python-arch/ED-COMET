#!/usr/bin/env python3
"""
Verify fairseq checkpoint and dictionary compatibility before conversion.

This script checks:
1. Actual vocabulary size in checkpoint
2. Special token indices used by fairseq
3. Dictionary size and mapping
4. Whether conversion will work
"""

import torch
import json
import sys
from pathlib import Path


def check_checkpoint(checkpoint_path):
    """Load and analyze fairseq checkpoint."""
    print(f"\n{'='*70}")
    print(f"Analyzing Fairseq Checkpoint")
    print(f"{'='*70}\n")

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"✓ Checkpoint loaded successfully\n")

    # Check embedding dimensions
    if 'encoder.embed_tokens.weight' in state_dict:
        encoder_embed_shape = state_dict['encoder.embed_tokens.weight'].shape
        print(f"Encoder embeddings shape: {encoder_embed_shape}")
        print(f"  Vocabulary size: {encoder_embed_shape[0]:,}")
        print(f"  Embedding dimension: {encoder_embed_shape[1]}")

    if 'decoder.embed_tokens.weight' in state_dict:
        decoder_embed_shape = state_dict['decoder.embed_tokens.weight'].shape
        print(f"\nDecoder embeddings shape: {decoder_embed_shape}")
        print(f"  Vocabulary size: {decoder_embed_shape[0]:,}")
        print(f"  Embedding dimension: {decoder_embed_shape[1]}")

    # Check if they match
    if encoder_embed_shape[0] == decoder_embed_shape[0]:
        print(f"\n✓ Encoder and decoder vocab sizes match: {encoder_embed_shape[0]:,}")
        vocab_size = encoder_embed_shape[0]
    else:
        print(f"\n⚠️  WARNING: Encoder and decoder vocab sizes don't match!")
        vocab_size = decoder_embed_shape[0]

    # Check checkpoint metadata
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"\nCheckpoint args:")
        if hasattr(args, 'task'):
            print(f"  Task: {args.task}")
        if hasattr(args, 'arch'):
            print(f"  Architecture: {args.arch}")

    return vocab_size, state_dict


def check_dictionary(dict_path):
    """Analyze fairseq dictionary."""
    print(f"\n{'='*70}")
    print(f"Analyzing Fairseq Dictionary")
    print(f"{'='*70}\n")

    print(f"Loading dictionary: {dict_path}")

    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    dict_size = len(lines)
    print(f"✓ Dictionary loaded")
    print(f"  Tokens in dict.txt: {dict_size:,}")

    # Show first 10 tokens
    print(f"\n  First 10 tokens:")
    for i, line in enumerate(lines[:10]):
        parts = line.split()
        token = parts[0] if parts else ""
        print(f"    {i}: {token}")

    # Check for special tokens in dict.txt
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    found_special = []
    for token in special_tokens:
        for line in lines:
            if line.split()[0] == token:
                found_special.append(token)
                break

    if found_special:
        print(f"\n  Special tokens found in dict.txt: {found_special}")
    else:
        print(f"\n  ℹ️  No special tokens in dict.txt (expected for fairseq)")
        print(f"     Fairseq reserves indices 0-4 for special tokens")

    return dict_size


def verify_conversion_compatibility(checkpoint_vocab_size, dict_size):
    """Check if conversion will work."""
    print(f"\n{'='*70}")
    print(f"Verifying Conversion Compatibility")
    print(f"{'='*70}\n")

    # Expected: dict_size + 5 special tokens = checkpoint_vocab_size
    expected_with_special = dict_size + 5

    print(f"Dictionary size (dict.txt):        {dict_size:,}")
    print(f"Expected special tokens:           5")
    print(f"Expected total vocabulary:         {expected_with_special:,}")
    print(f"Actual checkpoint vocabulary:      {checkpoint_vocab_size:,}")

    difference = checkpoint_vocab_size - expected_with_special

    if difference == 0:
        print(f"\n✅ PERFECT MATCH!")
        print(f"   Conversion should work correctly with 5 special tokens:")
        print(f"   - Index 0: <s>")
        print(f"   - Index 1: <pad>")
        print(f"   - Index 2: </s>")
        print(f"   - Index 3: <unk>")
        print(f"   - Index 4: <mask>")
        return True
    elif difference > 0:
        print(f"\n⚠️  MISMATCH: Checkpoint has {difference} MORE tokens than expected")
        print(f"   Possible reasons:")
        print(f"   - Additional special tokens beyond the standard 5")
        print(f"   - Dictionary is from a different checkpoint")
        return False
    else:
        print(f"\n⚠️  MISMATCH: Checkpoint has {abs(difference)} FEWER tokens than expected")
        print(f"   Possible reasons:")
        print(f"   - Fewer than 5 special tokens")
        print(f"   - Dictionary is from a different checkpoint")
        return False


def main():
    # Paths
    checkpoint_path = "/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/checkpoints/edbart_dapt/checkpoint_best.pt"
    dict_path = "/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/gdbart-bin/dict.txt"

    print(f"\n{'='*70}")
    print(f"FAIRSEQ TO HUGGINGFACE CONVERSION VERIFICATION")
    print(f"{'='*70}")

    # Check checkpoint
    result = check_checkpoint(checkpoint_path)
    if result is None:
        print("\n❌ Failed to analyze checkpoint. Exiting.")
        sys.exit(1)

    checkpoint_vocab_size, state_dict = result

    # Check dictionary
    dict_size = check_dictionary(dict_path)

    # Verify compatibility
    is_compatible = verify_conversion_compatibility(checkpoint_vocab_size, dict_size)

    # Final recommendation
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}\n")

    if is_compatible:
        print(f"✅ CONVERSION SHOULD WORK!")
        print(f"\nRecommended steps:")
        print(f"1. Run: python scripts/convert_custom_bart_checkpoint.py \\")
        print(f"       {checkpoint_path} \\")
        print(f"       ./checkpoints/bart_hf \\")
        print(f"       --hf_config facebook/bart-large")
        print(f"\n2. Run: python scripts/convert_fairseq_dict_to_hf_tokenizer.py \\")
        print(f"       {dict_path} \\")
        print(f"       ./checkpoints/bart_hf")
        print(f"\n3. Verify all files are created in ./checkpoints/bart_hf/")
    else:
        print(f"⚠️  CONVERSION MAY HAVE ISSUES!")
        print(f"\nYou may need to:")
        print(f"1. Verify you're using the correct dictionary")
        print(f"2. Check fairseq training logs for special token configuration")
        print(f"3. Consider staying with fairseq for training")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
