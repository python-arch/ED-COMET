#!/usr/bin/env python3
"""
Convert custom fairseq BART checkpoint to HuggingFace format.

This script handles fairseq BART checkpoints with custom vocabularies,
unlike the official conversion script which only works with standard BART models.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_ignore_keys_(state_dict):
    """Remove keys that should be ignored during conversion."""
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def infer_config_from_state_dict(state_dict, hf_config_name=None):
    """
    Infer BartConfig from the fairseq checkpoint state dict.

    Args:
        state_dict: The loaded fairseq model state dict
        hf_config_name: Optional HuggingFace config to use as base

    Returns:
        BartConfig with appropriate settings
    """
    # Get vocabulary size from embeddings
    vocab_size = state_dict['decoder.embed_tokens.weight'].shape[0]
    d_model = state_dict['decoder.embed_tokens.weight'].shape[1]

    # Count encoder/decoder layers
    encoder_layers = max([int(k.split('.')[2]) for k in state_dict.keys()
                          if k.startswith('encoder.layers.')]) + 1
    decoder_layers = max([int(k.split('.')[2]) for k in state_dict.keys()
                          if k.startswith('decoder.layers.')]) + 1

    # Get attention heads from first layer
    encoder_attention_heads = state_dict['encoder.layers.0.self_attn.k_proj.weight'].shape[0] // 64
    decoder_attention_heads = state_dict['decoder.layers.0.self_attn.k_proj.weight'].shape[0] // 64

    # Get FFN dimension
    encoder_ffn_dim = state_dict['encoder.layers.0.fc1.weight'].shape[0]
    decoder_ffn_dim = state_dict['decoder.layers.0.fc1.weight'].shape[0]

    logger.info(f"Inferred config from checkpoint:")
    logger.info(f"  vocab_size: {vocab_size}")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  encoder_layers: {encoder_layers}")
    logger.info(f"  decoder_layers: {decoder_layers}")
    logger.info(f"  encoder_attention_heads: {encoder_attention_heads}")
    logger.info(f"  decoder_attention_heads: {decoder_attention_heads}")
    logger.info(f"  encoder_ffn_dim: {encoder_ffn_dim}")
    logger.info(f"  decoder_ffn_dim: {decoder_ffn_dim}")

    # Create config based on inferred values
    if hf_config_name:
        # Start with base config and override sizes
        config = BartConfig.from_pretrained(hf_config_name)
        config.vocab_size = vocab_size
        config.d_model = d_model
        config.encoder_layers = encoder_layers
        config.decoder_layers = decoder_layers
        config.encoder_attention_heads = encoder_attention_heads
        config.decoder_attention_heads = decoder_attention_heads
        config.encoder_ffn_dim = encoder_ffn_dim
        config.decoder_ffn_dim = decoder_ffn_dim
    else:
        # Create new config from scratch
        config = BartConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
        )

    return config


@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_config_name=None):
    """
    Convert fairseq BART checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to fairseq checkpoint (.pt file)
        pytorch_dump_folder_path: Output directory for HuggingFace model
        hf_config_name: Optional HuggingFace config name to use as base
    """
    print(f"\n{'='*70}")
    print(f"Converting fairseq BART checkpoint to HuggingFace format")
    print(f"{'='*70}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Output directory: {pytorch_dump_folder_path}")
    print(f"Base config: {hf_config_name or 'None (creating from scratch)'}")
    print(f"{'='*70}\n")

    # Load the fairseq checkpoint
    print("Step 1: Loading fairseq checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"✓ Checkpoint loaded successfully")
    print(f"  Total keys in state dict: {len(state_dict)}")

    # Remove keys that should be ignored
    remove_ignore_keys_(state_dict)

    # Infer config from state dict
    print("\nStep 2: Inferring model configuration from checkpoint...")
    config = infer_config_from_state_dict(state_dict, hf_config_name)

    print(f"\n✓ Configuration inferred:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  encoder_layers: {config.encoder_layers}")
    print(f"  decoder_layers: {config.decoder_layers}")
    print(f"  encoder_attention_heads: {config.encoder_attention_heads}")
    print(f"  decoder_attention_heads: {config.decoder_attention_heads}")

    # Add shared embeddings (required by HuggingFace)
    if 'shared.weight' not in state_dict:
        state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']

    print("\nStep 3: Creating HuggingFace model with inferred config...")
    model = BartForConditionalGeneration(config)
    print(f"✓ Model created")
    print(f"  Model vocab size: {model.config.vocab_size}")
    print(f"  Embedding size: {model.model.shared.weight.shape}")

    # Load state dict into model
    print("\nStep 4: Loading checkpoint weights into model...")
    missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
    print(f"✓ Weights loaded")
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")

    # Set lm_head from shared embeddings
    if hasattr(model, "lm_head"):
        vocab_size, emb_size = model.model.shared.weight.shape
        model.lm_head.weight.data = model.model.shared.weight.data
        print(f"  LM head weight shape: {model.lm_head.weight.shape}")

    # Save the model
    print(f"\nStep 5: Saving model to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)

    # Verify the saved config
    import json
    saved_config_path = Path(pytorch_dump_folder_path) / "config.json"
    with open(saved_config_path, 'r') as f:
        saved_config = json.load(f)

    print(f"\n{'='*70}")
    print(f"✓ Conversion complete!")
    print(f"{'='*70}")
    print(f"Model saved to: {pytorch_dump_folder_path}")
    print(f"Saved vocab_size: {saved_config['vocab_size']}")
    print(f"Expected vocab_size: {config.vocab_size}")

    if saved_config['vocab_size'] != config.vocab_size:
        print(f"\n⚠️  WARNING: Saved vocab size doesn't match expected!")
        print(f"   This may cause issues when using the model.")
    else:
        print(f"\n✓ Vocabulary sizes match!")

    print(f"\n{'='*70}")
    print(f"IMPORTANT: Copy your custom tokenizer files to:")
    print(f"  {pytorch_dump_folder_path}/")
    print(f"Required files:")
    print(f"  - vocab.json")
    print(f"  - merges.txt")
    print(f"  - tokenizer_config.json (optional)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert custom fairseq BART checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "fairseq_path",
        type=str,
        help="Path to fairseq checkpoint (.pt file)"
    )
    parser.add_argument(
        "pytorch_dump_folder_path",
        type=str,
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--hf_config",
        default="facebook/bart-large",
        type=str,
        help="HuggingFace config to use as base (default: facebook/bart-large)"
    )

    args = parser.parse_args()
    convert_bart_checkpoint(
        args.fairseq_path,
        args.pytorch_dump_folder_path,
        hf_config_name=args.hf_config
    )
