#!/usr/bin/env python3
"""End-to-end SmolLM2 inference demo.

Loads weights from ``.npy`` files (downloaded by
``scripts/download_smollm2_weights.py``) and generates text.

Usage:
    python scripts/run_smollm2.py \\
        --weights ./weights \\
        --variant SmolLM2-135M \\
        --prompt \"Hello world\"
"""

import argparse
import os

import numpy as np

from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM


def load_state_dict(weights_dir: str) -> dict:
    """Load all ``.npy`` files from a directory into a state dict.

    Parameters
    ----------
    weights_dir : str
        Path to directory containing ``.npy`` files (as produced by
        ``scripts/download_smollm2_weights.py``).

    Returns
    -------
    state_dict : dict of str -> np.ndarray
        Mapping from HF weight name (dots replaced with underscores) to array.
    """
    state = {}
    if not os.path.exists(weights_dir):
        return state
    for fname in os.listdir(weights_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(weights_dir, fname)
        key = fname[:-4]  # remove .npy suffix, keep original name with dots
        arr = np.load(path, allow_pickle=False)
        state[key] = arr
    return state


def encode(prompt: str, vocab_size: int = 49152) -> np.ndarray:
    """Rudimentary character-level encoding (demo only — not a real tokeniser).

    For SmolLM2 use ``transformers.AutoTokenizer`` in production.

    Parameters
    ----------
    prompt : str
        Input text.
    vocab_size : int
        Upper bound for generated token IDs.

    Returns
    -------
    token_ids : np.ndarray, shape ``(1, seq)``, int32
    """
    # Simple character-level IDs for demo purposes
    ids = [min(ord(c) % vocab_size, vocab_size - 1) for c in prompt]
    if not ids:
        ids = [0]
    return np.array([ids], dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="SmolLM2 inference demo")
    parser.add_argument(
        "--weights", type=str, default="./weights",
        help="Directory with .npy weight files (default: ./weights)",
    )
    parser.add_argument(
        "--variant", type=str, default="SmolLM2-135M",
        help="Model variant (default: SmolLM2-135M)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello world",
        help="Prompt text (default: 'Hello world')",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10,
        help="Tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    args = parser.parse_args()

    # --- Config ---
    config = SmolLM2Config.from_pretrained(args.variant)
    print(f"Config: {config}")
    print(f"Loading weights from {args.weights}/ ...")

    # --- Weights ---
    state_dict = load_state_dict(args.weights)
    if len(state_dict) == 0:
        print("No .npy files found. Running with random weights for shape check.")
        state_dict = _make_random_weights(config)

    # --- Model ---
    model = SmolLM2ForCausalLM(config, state_dict)

    # --- Forward pass ---
    prompt_ids = encode(args.prompt, config.vocab_size)
    logits = model.forward(prompt_ids)
    print(f"Forward output shape: {logits.shape}")
    print(f"  (expected: (1, {prompt_ids.shape[1]}, {config.vocab_size}))")

    # --- Generate ---
    out = model.generate(
        prompt_ids.copy(),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Generated output shape: {out.shape}")
    print(f"  (expected: (1, {prompt_ids.shape[1] + args.max_new_tokens}))")
    print(f"Prompt tokens:  {prompt_ids.tolist()}")
    print(f"Output tokens:  {out.tolist()}")
    print("Done.")


def _make_random_weights(config) -> dict:
    """Create random weights for shape-checking."""
    n = config.n_embd
    v = config.vocab_size
    n_kv = config.n_kv_head
    n_h = config.n_head
    d_k = n // n_h
    n_layer = config.n_layer
    ffn = config.n_ffn

    weights = {
        "model.embed_tokens.weight": np.random.randn(v, n).astype(np.float32) * 0.02,
        "model.norm.weight": np.random.randn(n).astype(np.float32) * 0.02,
    }
    for i in range(n_layer):
        weights.update({
            f"model.layers.{i}.input_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
            f"model.layers.{i}.post_attention_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.q_proj.weight": np.random.randn(n_h * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.k_proj.weight": np.random.randn(n_kv * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.v_proj.weight": np.random.randn(n_kv * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.o_proj.weight": np.random.randn(n, n_h * d_k).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.gate_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.up_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.down_proj.weight": np.random.randn(n, ffn).astype(np.float32) * 0.02,
        })
    return weights


if __name__ == "__main__":
    main()
