#!/usr/bin/env python3
"""
GPT-2 inference demo — load weights from a .npz file, run forward and generate.

Usage:
    python scripts/run_gpt2.py [--weights path/to/gpt2_weights.npz] [--prompt N]
"""

import argparse
import numpy as np

from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model


def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference demo")
    parser.add_argument(
        "--weights", type=str, default="gpt2_weights.npz",
        help="Path to the .npz weights file (default: gpt2_weights.npz)",
    )
    parser.add_argument(
        "--prompt", type=int, nargs="+", default=[464, 4015],
        help="Prompt token IDs (default: [464, 4015])",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10,
        help="Number of tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    args = parser.parse_args()

    # --- Config ---
    config = GPT2Config()
    print(f"Config: {config}")
    print(f"Loading weights from {args.weights} ...")

    # --- Weights ---
    try:
        weights = np.load(args.weights)
    except FileNotFoundError:
        print(f"Weights file not found: {args.weights}")
        print("Run with random test weights instead.")
        weights = _make_random_weights(config)
        print("Using random weights (forward shapes only, output is random noise).")

    # --- Model ---
    model = GPT2Model(config, weights)

    # --- Forward pass ---
    prompt = np.array([args.prompt], dtype=np.int32)
    logits = model.forward(prompt)
    print(f"Forward output shape: {logits.shape}")
    print(f"  (expected: (1, {prompt.shape[1]}, {config.vocab_size}))")

    # --- Generate ---
    out = model.generate(
        prompt.copy(),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Generated output shape: {out.shape}")
    print(f"  (expected: (1, {prompt.shape[1] + args.max_new_tokens}))")
    print(f"Prompt tokens:  {prompt.tolist()}")
    print(f"Output tokens:  {out.tolist()}")

    print("Done.")


def _make_random_weights(config):
    """Create random weights for testing (matches the shape of real GPT-2 weights)."""
    n = config.n_embd
    v = config.vocab_size
    n_layer = config.n_layer
    weights = {
        "wte.weight": np.random.randn(v, n).astype(np.float32),
        "wpe.weight": np.random.randn(config.n_positions, n).astype(np.float32),
        "ln_f.weight": np.random.randn(n).astype(np.float32),
        "ln_f.bias": np.random.randn(n).astype(np.float32),
    }
    for i in range(n_layer):
        # GPT-2 uses Conv1D: weights stored as (in_features, out_features).
        # No transpose needed — gemm(hidden, w) uses w directly.
        weights.update({
            f"h.{i}.attn.c_attn.weight": np.random.randn(n, 3 * n).astype(np.float32),
            f"h.{i}.attn.c_attn.bias": np.random.randn(3 * n).astype(np.float32),
            f"h.{i}.attn.c_proj.weight": np.random.randn(n, n).astype(np.float32),
            f"h.{i}.attn.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.mlp.c_fc.weight": np.random.randn(n, 4 * n).astype(np.float32),
            f"h.{i}.mlp.c_fc.bias": np.random.randn(4 * n).astype(np.float32),
            f"h.{i}.mlp.c_proj.weight": np.random.randn(4 * n, n).astype(np.float32),
            f"h.{i}.mlp.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.bias": np.random.randn(n).astype(np.float32),
        })
    return weights


if __name__ == "__main__":
    main()
