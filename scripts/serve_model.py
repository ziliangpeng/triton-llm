#!/usr/bin/env python3
"""FastAPI HTTP server for SmolLM2 inference.

Usage on a GPU compute node:
    python scripts/serve_model.py --port 8000

From your local machine (after SSH tunnel is set up):
    python scripts/client.py --prompt "Hello world"

For fast testing without real weights:
    python scripts/serve_model.py --port 8000 --no-download
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("smollm2-server")

# ── Pydantic schemas ──────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text")
    max_tokens: int = Field(default=50, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None)
    stream: bool = Field(default=False, description="SSE streaming")


class StreamChoice(BaseModel):
    text: str
    finish_reason: str | None = None


class StreamResponse(BaseModel):
    choices: list[StreamChoice]


class ChatMessage(BaseModel):
    role: str = Field(..., description="One of: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=50, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None)
    stream: bool = Field(default=False, description="SSE streaming (not yet supported for chat)")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_seconds: float


class CompletionResponse(BaseModel):
    text: str
    tokens: list[int]
    usage: Usage


class ChatResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "length"


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-1"
    object: str = "chat.completion"
    choices: list[ChatResponseChoice]
    usage: Usage


# ── Globals (set during lifespan startup) ─────────────────────────────

model = None
tokenizer = None
model_variant = "SmolLM2-135M"
no_download = False
use_gpu = False
PORT = 8000  # default, overridden by CLI args in __main__


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


# ── Weights ───────────────────────────────────────────────────────────

def download_weights(variant: str) -> dict:
    """Download real SmolLM2 weights from HuggingFace, return numpy dict."""
    hf_name = f"HuggingFaceTB/{variant}"
    cache_dir = os.path.join(tempfile.gettempdir(), f"triton-llm-{variant}-weights")
    os.makedirs(cache_dir, exist_ok=True)

    marker = os.path.join(cache_dir, ".completed")
    if os.path.exists(marker):
        logger.info("  Loading cached weights...")
        return _load_dir(cache_dir)

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    logger.info(f"  Downloading {variant} from HuggingFace...")
    t0 = time.time()

    sf_path = hf_hub_download(
        repo_id=hf_name,
        filename="model.safetensors",
        cache_dir=os.path.join(cache_dir, "hf-cache"),
    )

    tensors = load_file(sf_path, device="cpu")
    logger.info(f"  Converting {len(tensors)} tensors to float32...")
    for key, tensor in tensors.items():
        arr = tensor.to(torch.float32).numpy()
        np.save(os.path.join(cache_dir, f"{key}.npy"), arr)

    with open(marker, "w") as f:
        f.write(f"download_time={time.time() - t0:.1f}s\n")

    dt = time.time() - t0
    logger.info(f"  Done ({len(tensors)} tensors, {dt:.1f}s)")
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
    d_head = n // cfg.num_attention_heads
    n_q = cfg.num_attention_heads * d_head
    n_kv = cfg.num_key_value_heads * d_head
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


def warmup(model, tokenizer):
    """Run dummy prefill + decode to trigger Triton JIT compilation.

    After warmup, no Triton compilation happens on the first real request.
    Uses generate() so internal _init_cache() is called properly.
    """
    ids = tokenizer.encode("a")
    token_ids = np.array([ids], dtype=np.int32)

    t0 = time.time()
    if use_gpu:
        _ = model.generate_gpu(token_ids, max_new_tokens=1, temperature=0.0)
    else:
        _ = model.generate(token_ids, max_new_tokens=1, temperature=0.0)
    dt = time.time() - t0
    logger.info(f"  Warmup prefill + 1 decode: {dt:.1f}s  (Triton compile all kernels)")


def format_chat_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a plain text prompt (base model, no chat template).

    Includes role prefixes so the model can distinguish system vs user vs assistant turns.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Capitalise role for readability in the plain-text prompt
        label = role.capitalize()
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    from smollm2_triton.config import SmolLM2Config
    from smollm2_triton.model import SmolLM2ForCausalLM

    logger.info(f"Loading {model_variant}...")

    # ── Config ──
    t0 = time.time()
    cfg = SmolLM2Config.from_pretrained(model_variant)
    logger.info(f"  Config loaded in {time.time() - t0:.1f}s")

    # ── Weights ──
    t0 = time.time()
    if no_download:
        weights = random_weights(cfg)
        logger.info(f"  Random weights created in {time.time() - t0:.1f}s")
    else:
        try:
            weights = download_weights(model_variant)
        except Exception as e:
            logger.warning(f"  Download failed ({e}), falling back to random weights")
            weights = random_weights(cfg)
        logger.info(f"  Weights loaded in {time.time() - t0:.1f}s")

    # ── Model creation ──
    t0 = time.time()
    model = SmolLM2ForCausalLM(cfg, weights)
    logger.info(f"  Model created in {time.time() - t0:.1f}s  (numpy prep + weight transpose)")

    # ── Tokenizer ──
    try:
        tokenizer = load_tokenizer(model_variant)
    except Exception as e:
        logger.warning(f"  Tokenizer failed ({e})")

    # ── Warmup ──
    if tokenizer is not None and model is not None:
        t0 = time.time()
        warmup(model, tokenizer)
        logger.info(f"  Warmup total: {time.time() - t0:.1f}s")
    else:
        logger.warning("  Skipping warmup (model or tokenizer not available)")

    logger.info(f"  Server ready on port {PORT}")
    yield


app = FastAPI(title="SmolLM2 Triton Inference", version="0.2.0", lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_variant, "loaded": model is not None}


def _generate(req_prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int | None):
    """Shared generation logic for both /v1/completions and /v1/chat/completions."""
    token_ids = encode(req_prompt)
    seq_len = token_ids.shape[1]
    _validate_request(seq_len, max_tokens, model.config.n_positions)

    if seed is not None:
        np.random.seed(seed)

    t0 = time.time()
    if use_gpu:
        out = model.generate_gpu(token_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    else:
        out = model.generate(token_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    dt = time.time() - t0

    new_ids = out[0, seq_len:].tolist()
    new_text = decode(new_ids)

    return new_text, new_ids, seq_len, dt


def _validate_request(seq_len, max_tokens, n_positions):
    """Shared input validation for both sync and streaming generation.

    Raises HTTPException if any check fails.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if tokenizer is None:
        raise HTTPException(503, "Tokenizer not loaded")
    if seq_len == 0:
        raise HTTPException(400, "Empty prompt")
    if seq_len + max_tokens > n_positions:
        raise HTTPException(400,
                            f"Sequence too long: {seq_len} prompt + {max_tokens} new = "
                            f"{seq_len + max_tokens}, max = {n_positions}")


async def _stream_generate(req_prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int | None):
    """Async generator that yields SSE-formatted token chunks in **real time**.

    Runs ``model.generate_stream()`` in a background thread and bridges
    results to the async SSE generator via an ``asyncio.Queue``, so each
    token is sent to the client as soon as its decode step finishes.

    Followed by a usage stats chunk (with TTFT, TPOT), then ``[DONE]``.
    """
    token_ids = encode(req_prompt)
    seq_len = token_ids.shape[1]

    if seed is not None:
        np.random.seed(seed)

    # Queue to bridge sync generate_stream() -> async SSE
    # Use 0 (unbounded) for GPU path where tokens arrive in burst;
    # the consumer drains at steady rate via SSE chunks.
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)
    loop = asyncio.get_running_loop()

    def _run():
        """Runs in a background thread — consumes the sync generator."""
        try:
            if use_gpu:
                # GPU path: generate all tokens at once, yield one by one
                t_prefill = time.time()
                out_gpu = model.generate_gpu(
                    token_ids, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k,
                )
                t_after = time.time()
                # First token = prefill; rest = decode (approximate per-token)
                new_tokens = out_gpu[0, seq_len:].tolist()
                per_step = (t_after - t_prefill) / max(len(new_tokens), 1)
                for i, tid in enumerate(new_tokens):
                    step_time = t_after - t_prefill if i == 0 else per_step
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("token", tid, step_time)
                    )
            else:
                for token_id, step_time in model.generate_stream(
                    token_ids, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k,
                ):
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("token", token_id, step_time)
                    )
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None, None))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e), None))
            logger.exception("generate_stream error")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    step_times: list[float] = []

    while True:
        msg_type, payload, step_time = await queue.get()
        if msg_type == "token":
            token_text = decode([payload])
            chunk = {
                "choices": [{"text": token_text, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            step_times.append(step_time)
        elif msg_type == "done":
            break
        elif msg_type == "error":
            yield f"data: {json.dumps({'error': payload})}\n\n"
            return

    # Final chunk with finish_reason
    final_chunk = {
        "choices": [{"text": "", "finish_reason": "length"}],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    # Usage / performance stats chunk
    ttft = step_times[0]
    num_decodes = len(step_times)
    total_time = sum(step_times)
    tpot = (sum(step_times[1:]) / max(num_decodes - 1, 1)
            if num_decodes > 1 else 0.0)
    tps = round(num_decodes / total_time, 1) if total_time > 0 else 0.0

    usage_chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": seq_len,
            "completion_tokens": num_decodes,
            "total_tokens": seq_len + num_decodes,
            "time_seconds": round(total_time, 3),
            "ttft_ms": round(ttft * 1000, 1),
            "tpot_ms": round(tpot * 1000, 1),
            "tokens_per_second": tps,
        },
    }
    yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    # Validate before StreamingResponse — HTTPException must be caught by FastAPI
    # in the route handler, not inside the async generator.
    if req.stream and (model is None or tokenizer is None):
        raise HTTPException(503, "Model or tokenizer not loaded")

    if req.stream:
        token_ids = encode(req.prompt)
        seq_len = token_ids.shape[1]
        _validate_request(seq_len, req.max_tokens, model.config.n_positions)
        return StreamingResponse(
            _stream_generate(req.prompt, req.max_tokens, req.temperature, req.top_k, req.seed),
            media_type="text/event-stream",
        )
    new_text, new_ids, seq_len, dt = _generate(
        req.prompt, req.max_tokens, req.temperature, req.top_k, req.seed
    )
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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(400, "Chat streaming not yet supported; use /v1/completions with stream=true")
    prompt = format_chat_prompt([m.model_dump() for m in req.messages])
    new_text, new_ids, seq_len, dt = _generate(
        prompt, req.max_tokens, req.temperature, req.top_k, req.seed
    )
    return ChatCompletionResponse(
        usage=Usage(
            prompt_tokens=seq_len,
            completion_tokens=len(new_ids),
            total_tokens=seq_len + len(new_ids),
            time_seconds=round(dt, 3),
        ),
        choices=[
            ChatResponseChoice(
                message=ChatMessage(role="assistant", content=new_text),
                finish_reason="length",
            )
        ],
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
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU-resident inference path (7-8x faster)")
    p.add_argument("--log-level", type=str, default="info",
                   choices=["debug", "info", "warning", "error"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(message)s")
    model_variant = args.model
    no_download = args.no_download
    use_gpu = args.gpu
    PORT = args.port
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
