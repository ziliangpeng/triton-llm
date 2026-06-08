#!/usr/bin/env python3
"""FastAPI HTTP server for SmolLM2 inference with real weights.

Usage on GPU compute node:
    python scripts/serve_model.py --port 8000 --no-download

From local machine (after SSH tunnel is set up):
    python scripts/client.py --prompt "Hello world"

To set up SSH tunnel (from maccai):
    ssh -J gcp5 -L 8000:localhost:8000 -N gcp5-h100-0-9 &
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("smollm2-server")

# ── Schemas ───────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input text")
    max_tokens: int = Field(default=50, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_seconds: float


class CompletionResponse(BaseModel):
    text: str
    tokens: list[int]
    usage: Usage


# ── Global state ──────────────────────────────────────────────────────

# Filled during lifespan startup
model = None
tokenizer = None
model_variant = "SmolLM2-135M"
no_download = False  # set from CLI


# ── Tokenizer ─────────────────────────────────────────────────────────

def load_tokenizer(variant: str):
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer...")
    t = AutoTokenizer.from_pretrained(f"HuggingFaceTB/{variant}")
    logger.info(f"  Tokenizer loaded (vocab_size={t.vocab_size})")
    return t


def encode(text: str) -> np.ndarray:
    ids = tokenizer.encode(text)
    return np.array([ids], dtype=np.int32)


def decode(ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


# ── Weights ────────────────────────────────────────────────────────────

def download_weights(variant: str) -> dict:
    """Download real SmolLM2 weights from HuggingFace, return numpy dict."""
    hf_name = f"HuggingFaceTB/{variant}"
    cache_dir = f"/tmp/{variant}-weights"
    os.makedirs(cache_dir, exist_ok=True)

    marker = os.path.join(cache_dir, ".completed")
    if os.path.exists(marker):
        logger.info("  Loading cached weights...")
        return _load_dir(cache_dir)

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    logger.info(f"  Downloading {variant} from HuggingFace...")
    t0 = time.time()

    sf_path = hf_hub_download(
        repo_id=hf_name,
        filename="model.safetensors",
        cache_dir=os.path.join(cache_dir, "hf-cache"),
    )

    with safe_open(sf_path, framework="np", device="cpu") as f:
        keys = f.keys()
        logger.info(f"  Converting {len(keys)} tensors to float32...")
        for key in keys:
            arr = f.get_tensor(key)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            np.save(os.path.join(cache_dir, f"{key}.npy"), arr)

    with open(marker, "w") as f:
        f.write(f"download_time={time.time() - t0:.1f}s\n")

    dt = time.time() - t0
    logger.info(f"  Done ({len(keys)} tensors, {dt:.1f}s)")
    return _load_dir(cache_dir)


def _load_dir(cache_dir: str) -> dict:
    weights = {}
    for fname in os.listdir(cache_dir):
        if fname.endswith(".npy"):
            weights[fname[:-4]] = np.load(os.path.join(cache_dir, fname))
    return weights


def random_weights(cfg) -> dict:
    """Generate random weights for testing (output will be gibberish)."""
    n = cfg.hidden_size
    n_q = cfg.num_attention_heads * 64
    n_kv = cfg.num_key_value_heads * 64
    ffn = cfg.intermediate_size
    np.random.seed(42)
    w = {}
    w["model.embed_tokens.weight"] = np.random.randn(cfg.vocab_size, n).astype(np.float32)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        w[f"{p}self_attn.q_proj.weight"] = np.random.randn(n_q, n).astype(np.float32)
        w[f"{p}self_attn.k_proj.weight"] = np.random.randn(n_kv, n).astype(np.float32)
        w[f"{p}self_attn.v_proj.weight"] = np.random.randn(n_kv, n).astype(np.float32)
        w[f"{p}self_attn.o_proj.weight"] = np.random.randn(n, n_q).astype(np.float32)
        w[f"{p}mlp.gate_proj.weight"] = np.random.randn(ffn, n).astype(np.float32)
        w[f"{p}mlp.up_proj.weight"] = np.random.randn(ffn, n).astype(np.float32)
        w[f"{p}mlp.down_proj.weight"] = np.random.randn(n, ffn).astype(np.float32)
        w[f"{p}input_layernorm.weight"] = np.random.randn(n).astype(np.float32)
        w[f"{p}post_attention_layernorm.weight"] = np.random.randn(n).astype(np.float32)
    w["model.norm.weight"] = np.random.randn(n).astype(np.float32)
    w["lm_head.weight"] = np.random.randn(cfg.vocab_size, n).astype(np.float32)
    logger.warning("  Using RANDOM weights — output will be gibberish!")
    return w


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    from smollm2_triton.config import SmolLM2Config
    from smollm2_triton.model import SmolLM2ForCausalLM

    logger.info(f"Loading {model_variant}...")

    # 1) Config
    cfg = SmolLM2Config.from_pretrained(model_variant)

    # 2) Weights
    if no_download:
        weights = random_weights(cfg)
    else:
        try:
            weights = download_weights(model_variant)
        except Exception as e:
            logger.warning(f"  Download failed ({e}), falling back to random weights")
            weights = random_weights(cfg)

    # 3) Model
    t0 = time.time()
    model = SmolLM2ForCausalLM(cfg, weights)
    logger.info(f"  Model created in {time.time() - t0:.1f}s")

    # 4) Tokenizer (best-effort — server works without it)
    try:
        tokenizer = load_tokenizer(model_variant)
    except Exception as e:
        logger.warning(f"  Tokenizer failed ({e}), prompt will be shown as token IDs")

    logger.info(f"  Server ready on port {PORT}")
    yield


app = FastAPI(title="SmolLM2 Triton Inference", version="0.2.0", lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_variant, "loaded": model is not None}


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    token_ids = encode(req.prompt)
    seq_len = token_ids.shape[1]
    if seq_len == 0:
        raise HTTPException(400, "Empty prompt")

    if req.seed is not None:
        np.random.seed(req.seed)

    t0 = time.time()
    out = model.generate(
        token_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    dt = time.time() - t0

    new_ids = out[0, seq_len:].tolist()
    new_text = decode(new_ids)

    return CompletionResponse(
        text=new_text,
        tokens=new_ids,
        usage=Usage(
            prompt_tokens=seq_len,
            completion_tokens=len(new_ids),
            total_tokens=seq_len + len(new_ids),
            time_seconds=round(dt, 3),
        ),
    )


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SmolLM2 Triton HTTP server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--model", type=str, default="SmolLM2-135M",
                   choices=["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"])
    p.add_argument("--no-download", action="store_true",
                   help="Skip real weights, use random (fast, gibberish output)")
    p.add_argument("--log-level", type=str, default="info",
                   choices=["debug", "info", "warning", "error"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(message)s")
    model_variant = args.model
    no_download = args.no_download
    PORT = args.port
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
