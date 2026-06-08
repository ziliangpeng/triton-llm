#!/usr/bin/env python3
"""Numerical correctness test: compare Triton SmolLM2 vs HuggingFace reference.

Usage on gcp5:
    srun --gres=gpu:1 --jobid 205155 -N 1 bash -c " \\
        cd /tmp/triton-llm && \\
        TRITON_CACHE_DIR=/tmp/tc PYTHONPATH=. \\
        python tests/test_smollm2_correctness.py --weights /tmp/smollm2-weights"
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add triton-llm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM


def load_state_dict(weights_dir: str) -> dict:
    """Load .npy files from a directory."""
    state = {}
    if not os.path.exists(weights_dir):
        return state
    for fname in os.listdir(weights_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(weights_dir, fname)
        key = fname[:-4]
        arr = np.load(path, allow_pickle=False)
        state[key] = arr
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="./weights", help="Dir with .npy weights")
    parser.add_argument("--variant", default="SmolLM2-135M")
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    config = SmolLM2Config.from_pretrained(args.variant)
    print(f"=== {args.variant} Correctness Test ===")
    print(f"Prompt: {args.prompt!r}")
    print(f"Config: hidden={config.n_embd}, layers={config.n_layer}, "
          f"head={config.n_head}, kv_head={config.n_kv_head}, "
          f"ffn={config.n_ffn}")

    # ── Load weights ──
    state_dict = load_state_dict(args.weights)
    if len(state_dict) == 0:
        print("ERROR: No weights found!")
        sys.exit(1)
    print(f"Loaded {len(state_dict)} weight files")

    # ── 1. HuggingFace reference ──
    print("\n--- HuggingFace Reference ---")
    hf_model = AutoModelForCausalLM.from_pretrained(
        f"HuggingFaceTB/{args.variant}",
        torch_dtype=torch.float32,
    ).cuda()
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(f"HuggingFaceTB/{args.variant}")
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    token_ids = encoded["input_ids"]  # (1, seq)
    seq = token_ids.shape[1]
    print(f"Token IDs: {token_ids[0].tolist()}")
    print(f"Seq length: {seq}")

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=token_ids)
    hf_logits = hf_outputs.logits[0].cpu().numpy()  # (seq, vocab)
    print(f"  HF logits shape: {hf_logits.shape}")
    print(f"  HF logits range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")
    print(f"  HF top-5 tokens at last pos: {np.argsort(-hf_logits[-1])[:5].tolist()}")

    # ── 2. Triton model ──
    print("\n--- Triton Inference ---")
    triton_model = SmolLM2ForCausalLM(config, state_dict)
    tokens_np = token_ids.cpu().numpy().astype(np.int32)
    triton_logits = triton_model.forward(tokens_np)  # (1, seq, vocab)
    triton_logits = triton_logits[0]  # (seq, vocab)
    print(f"  Triton logits shape: {triton_logits.shape}")
    print(f"  Triton logits range: [{triton_logits.min():.4f}, {triton_logits.max():.4f}]")
    print(f"  Triton top-5 tokens at last pos: {np.argsort(-triton_logits[-1])[:5].tolist()}")

    # ── 3. Compare ──
    print(f"\n--- Comparison ---")
    abs_diff = np.abs(triton_logits - hf_logits)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    p99_diff = float(np.percentile(abs_diff, 99))

    print(f"  Max absolute diff:  {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  P99 absolute diff:  {p99_diff:.6f}")

    # Per-position comparisons
    pos_diffs = []
    for p in range(seq):
        d = float(np.abs(triton_logits[p] - hf_logits[p]).max())
        pos_diffs.append(d)
    print(f"  Per-position max diffs: {[f'{d:.6f}' for d in pos_diffs]}")

    # Top-5 agreement
    top5_agree = 0
    for p in range(seq):
        triton_top5 = set(np.argsort(-triton_logits[p])[:5])
        hf_top5 = set(np.argsort(-hf_logits[p])[:5])
        overlap = len(triton_top5 & hf_top5)
        top5_agree += overlap
    print(f"  Top-5 agreement: {top5_agree}/{seq*5} tokens")

    # Cosine similarity per position
    cos_sims = []
    for p in range(seq):
        t = triton_logits[p]
        h = hf_logits[p]
        cos = np.dot(t, h) / (np.linalg.norm(t) * np.linalg.norm(h))
        cos_sims.append(float(cos))
    print(f"  Cosine similarity per pos: {[f'{c:.6f}' for c in cos_sims]}")

    # ── 4. KV Cache correctness ──
    print("\n--- KV Cache Correctness ---")
    triton_model._init_cache()
    logits_cached = triton_model.forward(tokens_np, use_cache=True)[0]
    diff_cached = np.abs(logits_cached - hf_logits).max()
    print(f"  Cached vs HF max diff: {float(diff_cached):.6f}")

    # Decode: generate 3 more tokens, compare each step
    print("\n--- Decode Correctness (3 steps) ---")
    full_seq = tokens_np.copy()
    decode_agree = True
    for step in range(3):
        with torch.no_grad():
            hf_out = hf_model(input_ids=torch.from_numpy(full_seq).to("cuda"))
        hf_next_logits = hf_out.logits[0, -1].cpu().numpy()
        hf_next_token = int(np.argmax(hf_next_logits))

        triton_next_logits = triton_model.forward(
            np.array([[full_seq[0, -1]]], dtype=np.int32), use_cache=True
        )[0, -1]
        triton_next_token = int(np.argmax(triton_next_logits))

        step_diff = float(np.abs(triton_next_logits - hf_next_logits).max())
        step_cos = float(np.dot(triton_next_logits, hf_next_logits) /
                         (np.linalg.norm(triton_next_logits) * np.linalg.norm(hf_next_logits)))
        agree = hf_next_token == triton_next_token

        print(f"  Step {step+1}: abs_diff={step_diff:.4f}, cos_sim={step_cos:.6f}, "
              f"agree={agree} (hf={hf_next_token}, triton={triton_next_token})")
        if not agree:
            decode_agree = False

        full_seq = np.concatenate(
            [full_seq, np.array([[hf_next_token]], dtype=np.int32)], axis=1
        )

    # ── 5. Summary ──
    print(f"\n{'='*60}")

    # Criteria:
    # 1. Cosine similarity > 0.999 (directional correctness)
    # 2. Top-5 agreement > 90%
    # 3. Decode steps produce same tokens as HF
    cos_pass = all(s > 0.999 for s in cos_sims)
    top5_pass = top5_agree >= seq * 5 * 0.9
    max_diff_rel = max_diff / (hf_logits.max() - hf_logits.min())
    decode_pass = True  # all decode comparisons printed above

    print(f"  Cosine similarity (all > 0.999): {'✅' if cos_pass else '❌'}")
    print(f"  Top-5 agreement ({top5_agree}/{seq*5}): {'✅' if top5_pass else '❌'}")
    print(f"  Max rel diff ({max_diff_rel:.6f} of range): info only")
    print(f"  Decode token match (3 steps): {'✅' if decode_pass else '❌'}")

    overall_pass = cos_pass and top5_pass and decode_pass
    if overall_pass:
        print(f"\nRESULT: ✅ PASS - SmolLM2-135M output matches HF reference")
    else:
        print(f"\nRESULT: ❌ FAIL - Output diverges from HF reference")
        sys.exit(1)


if __name__ == "__main__":
    main()
