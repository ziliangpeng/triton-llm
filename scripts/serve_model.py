#!/usr/bin/env python3
"""FastAPI HTTP server for SmolLM2 inference.

Usage on GPU compute node:
    python scripts/serve_model.py --port 8000

Then from maccai:
    # Tunnel through login node to compute node
    ssh -J gcp5 -L 8000:localhost:8000 -N gcp5-h100-0-9 &

    # Or via login node proxy:
    ssh -L 8000:localhost:8000 -N gcp5 &

    curl http://localhost:8000/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "The capital of France is", "max_tokens": 20}'
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so model imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM

app = FastAPI(title="SmolLM2 Triton Inference", version="0.1.0")

# ── Globals (set during startup) ──────────────────────────────────────
model: SmolLM2ForCausalLM | None = None
config: SmolLM2Config | None = None
MODEL_VARIANT = "SmolLM2-135M"
NO_DOWNLOAD = False


# ── Pydantic schemas ──────────────────────────────────────────────────
class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(default=50, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    seed: int | None = Field(default=None, description="Random seed for deterministic sampling")


class CompletionResponse(BaseModel):
    text: str
    tokens: list[int]
    usage: dict


# ── Tokenizer (simple BPE for SmolLM2) ───────────────────────────────
# SmolLM2 uses the same tokenizer as Llama 3 (tiktoken-based).
# For the prototype, we download the tokenizer from HF at startup.
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        _tokenizer = t
        return t
    except Exception as e:
        print(f"WARNING: Could not load tokenizer: {e}")
        return None


def encode(text: str) -> np.ndarray:
    tok = _get_tokenizer()
    if tok is None:
        return np.array([[1, 284, 948]], dtype=np.int32)  # dummy
    ids = tok.encode(text)
    return np.array([ids], dtype=np.int32)


def decode(ids: list[int]) -> str:
    tok = _get_tokenizer()
    if tok is None:
        return str(ids)
    return tok.decode(ids)


# ── Weight download ──────────────────────────────────────────────────
def download_weights(variant: str, cache_dir: str = "/tmp/smollm2-weights") -> dict:
    """Download real SmolLM2 weights from HuggingFace."""
    hf_name = f"HuggingFaceTB/{variant}"
    os.makedirs(cache_dir, exist_ok=True)

    # Check if already cached
    if os.path.exists(os.path.join(cache_dir, "lm_head.weight.npy")):
        print(f"Loading cached weights from {cache_dir}")
        return _load_cached_weights(cache_dir)

    print(f"Downloading {variant} weights from HuggingFace...")
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # Download model.safetensors
    safetensors_path = hf_hub_download(
        repo_id=hf_name,
        filename="model.safetensors",
        cache_dir=os.path.join(cache_dir, "hf-cache"),
    )

    # Convert to .npy files
    with safe_open(safetensors_path, framework="np", device="cpu") as f:
        keys = f.keys()
        print(f"  Converting {len(keys)} tensors to .npy...")
        for key in keys:
            arr = f.get_tensor(key)
            # bfloat16 → float32
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            elif arr.dtype == np.dtype("bfloat16"):
                arr = arr.astype(np.float32)
            np.save(os.path.join(cache_dir, f"{key}.npy"), arr)

    print(f"  Saved {len(keys)} .npy files to {cache_dir}")
    return _load_cached_weights(cache_dir)


def _load_cached_weights(cache_dir: str) -> dict:
    """Load all .npy files into a dict."""
    weights = {}
    for fname in os.listdir(cache_dir):
        if fname.endswith(".npy"):
            key = fname[:-4]
            weights[key] = np.load(os.path.join(cache_dir, fname))
    return weights


# ── Startup event ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global model, config
    config = SmolLM2Config()

    print(f"Loading {MODEL_VARIANT}...")
    t0 = time.time()

    # Try real weights, fall back to random
    if NO_DOWNLOAD:
        print("  Using random weights (--no-download)")
        weights = _make_random_weights(config)
    else:
        try:
            weights = download_weights(MODEL_VARIANT)
            print(f"  Loaded {len(weights)} weight tensors")
        except Exception as e:
            print(f"  WARNING: Weight download failed ({e}), using random weights")
            weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    dt = time.time() - t0
    print(f"  Model loaded in {dt:.1f}s")

    # Warm up the tokenizer
    _get_tokenizer()
    print(f"  Server ready on port {PORT}")


# ── API endpoints ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_VARIANT, "loaded": model is not None}


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Encode
    token_ids = encode(req.prompt)
    seq_len = token_ids.shape[1]

    if seq_len == 0:
        raise HTTPException(400, "Prompt must not be empty")

    # Set seed if requested
    if req.seed is not None:
        np.random.seed(req.seed)

    # Generate
    t0 = time.time()
    out = model.generate(
        token_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    dt = time.time() - t0

    # Decode only the newly generated tokens
    new_ids = out[0, seq_len:].tolist()
    new_text = decode(new_ids)

    return CompletionResponse(
        text=new_text,
        tokens=new_ids,
        usage={
            "prompt_tokens": seq_len,
            "completion_tokens": len(new_ids),
            "total_tokens": seq_len + len(new_ids),
            "time_seconds": round(dt, 3),
        },
    )


# ── Random weight fallback ───────────────────────────────────────────
def _make_random_weights(cfg):
    np.random.seed(42)
    n = cfg.hidden_size
    n_q = cfg.num_attention_heads * 64
    n_kv = cfg.num_key_value_heads * 64
    ffn = cfg.intermediate_size
    w = {}
    w["model.embed_tokens.weight"] = np.random.randn(cfg.vocab_size, n).astype(np.float32)
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."
        w[f"{prefix}self_attn.q_proj.weight"] = np.random.randn(n_q, n).astype(np.float32)
        w[f"{prefix}self_attn.k_proj.weight"] = np.random.randn(n_kv, n).astype(np.float32)
        w[f"{prefix}self_attn.v_proj.weight"] = np.random.randn(n_kv, n).astype(np.float32)
        w[f"{prefix}self_attn.o_proj.weight"] = np.random.randn(n, n_q).astype(np.float32)
        w[f"{prefix}mlp.gate_proj.weight"] = np.random.randn(ffn, n).astype(np.float32)
        w[f"{prefix}mlp.up_proj.weight"] = np.random.randn(ffn, n).astype(np.float32)
        w[f"{prefix}mlp.down_proj.weight"] = np.random.randn(n, ffn).astype(np.float32)
        w[f"{prefix}input_layernorm.weight"] = np.random.randn(n).astype(np.float32)
        w[f"{prefix}post_attention_layernorm.weight"] = np.random.randn(n).astype(np.float32)
    w["model.norm.weight"] = np.random.randn(n).astype(np.float32)
    w["lm_head.weight"] = np.random.randn(cfg.vocab_size, n).astype(np.float32)
    return w


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SmolLM2 Triton HTTP server")
    p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Host to bind. Use 0.0.0.0 only behind SSH tunnel (default: 127.0.0.1)")
    p.add_argument("--model", type=str, default="SmolLM2-135M",
                   choices=["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"],
                   help="Model variant (default: SmolLM2-135M)")
    p.add_argument("--no-download", action="store_true",
                   help="Skip real weight download, use random weights")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_VARIANT = args.model
    PORT = args.port
    NO_DOWNLOAD = args.no_download

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
