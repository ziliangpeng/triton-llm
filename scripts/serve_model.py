#!/usr/bin/env python3
"""FastAPI HTTP server for SmolLM2 inference (GPU only).

Usage:
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
    stream: bool = Field(default=False, description="SSE streaming")


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

PORT = 8000  # default, overridden by CLI args in __main__


def is_gpt2_variant(variant: str) -> bool:
    return variant in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}


def hf_repo_id(variant: str) -> str:
    if is_gpt2_variant(variant):
        return f"openai-community/{variant}"
    return f"HuggingFaceTB/{variant}"


# ── Tokenizer ─────────────────────────────────────────────────────────

def load_tokenizer(variant: str):
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer...")
    t = AutoTokenizer.from_pretrained(hf_repo_id(variant))
    logger.info(f"  Tokenizer loaded (vocab_size={t.vocab_size})")
    return t


def encode(text: str) -> np.ndarray:
    ids = tokenizer.encode(text)
    return np.array([ids], dtype=np.int32)


def decode(ids: list[int]) -> str:
    """Decode token IDs to text, stripping leading role prefixes.

    EOS tokens (``<|im_end|>``) are stripped by ``skip_special_tokens=True``.
    Instruct models sometimes generate the role prefix as literal text in the
    first token, so we strip those leading markers.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for prefix in ("assistant\n", "user\n", "system\n"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


# ── Weights ───────────────────────────────────────────────────────────

def download_weights(variant: str) -> dict:
    """Download real model weights from HuggingFace, return numpy dict."""
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
        repo_id=hf_repo_id(variant),
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
    np.random.seed(42)
    if hasattr(cfg, "hidden_size"):
        n = cfg.hidden_size
        d_head = n // cfg.num_attention_heads
        n_q = cfg.num_attention_heads * d_head
        n_kv = cfg.num_key_value_heads * d_head
        ffn = cfg.intermediate_size
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

    # GPT-2 config layout
    n = cfg.n_embd
    v = cfg.vocab_size
    weights = {
        "wte.weight": np.random.randn(v, n).astype(np.float32),
        "wpe.weight": np.random.randn(cfg.n_positions, n).astype(np.float32),
        "ln_f.weight": np.random.randn(n).astype(np.float32),
        "ln_f.bias": np.random.randn(n).astype(np.float32),
    }
    for i in range(cfg.n_layer):
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
    logger.warning("  Using RANDOM weights — output will be gibberish!")
    return weights


def warmup(model, tokenizer):
    """Run dummy prefill + decode to trigger Triton JIT compilation.

    After warmup, no Triton compilation happens on the first real request.
    Uses generate_gpu() so internal _init_cache() is called properly.
    """
    ids = tokenizer.encode("a")
    token_ids = np.array([ids], dtype=np.int32)

    t0 = time.time()
    _ = model.generate_gpu(token_ids, max_new_tokens=1, temperature=0.0)
    dt = time.time() - t0
    logger.info(f"  Warmup prefill + 1 decode: {dt:.1f}s  (Triton compile all kernels)")


def format_chat_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a prompt using the tokenizer's chat template.

    For Instruct models (e.g. SmolLM2-135M-Instruct), uses the proper ChatML-style
    template with ``<|im_start|>`` tags. For base models, falls back to plain text.
    """
    if hasattr(tokenizer, "apply_chat_template") and "Instruct" in model_variant:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    # Fallback: plain text for base models
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        label = role.capitalize()
        parts.append(f"{label}: {content}")
    return "\n".join(parts) + "\nAssistant:"


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    logger.info(f"Loading {model_variant}...")

    # ── Config / model class selection ──
    t0 = time.time()
    if is_gpt2_variant(model_variant):
        from gpt2_triton.config import GPT2Config
        from gpt2_triton.model import GPT2Model

        cfg = GPT2Config.from_pretrained(model_variant)
        ModelClass = GPT2Model
    else:
        from smollm2_triton.config import SmolLM2Config
        from smollm2_triton.model import SmolLM2ForCausalLM

        cfg = SmolLM2Config.from_pretrained(model_variant)
        ModelClass = SmolLM2ForCausalLM
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
    model = ModelClass(cfg, weights)
    logger.info(f"  Model created in {time.time() - t0:.1f}s")

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


def max_context_positions(model_obj) -> int:
    cfg = model_obj.config
    if hasattr(cfg, "n_positions"):
        return cfg.n_positions
    return cfg.max_position_embeddings


def _generate(req_prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int | None):
    """Shared generation logic for both /v1/completions and /v1/chat/completions."""
    token_ids = encode(req_prompt)
    seq_len = token_ids.shape[1]
    _validate_request(seq_len, max_tokens, max_context_positions(model))

    if seed is not None:
        np.random.seed(seed)

    t0 = time.time()
    out = model.generate_gpu(token_ids, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k,
                             eos_token_id=tokenizer.eos_token_id)
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

    Runs ``model.generate_stream_gpu()`` in a background thread and bridges
    results to the async SSE generator via an ``asyncio.Queue``, so each
    token is sent to the client as soon as its decode step finishes.

    Followed by a usage stats chunk (with TTFT, TPOT), then ``[DONE]``.
    """
    token_ids = encode(req_prompt)
    seq_len = token_ids.shape[1]

    if seed is not None:
        np.random.seed(seed)

    # Queue to bridge sync generate_stream_gpu() -> async SSE
    # Use 0 (unbounded) so the consumer can drain at its own pace.
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)
    loop = asyncio.get_running_loop()

    def _run():
        """Runs in a background thread — consumes the sync generator."""
        try:
            for token_id, step_time in model.generate_stream_gpu(
                token_ids, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
                eos_token_id=tokenizer.eos_token_id,
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
    all_ids: list[int] = []
    prev_text = ""

    while True:
        msg_type, payload, step_time = await queue.get()
        if msg_type == "token":
            all_ids.append(payload)
            token_text = decode(all_ids)
            # Extract only the new characters since last token
            if token_text.startswith(prev_text):
                delta_text = token_text[len(prev_text):]
            else:
                delta_text = token_text
            prev_text = token_text
            chunk = {
                "choices": [{"text": delta_text, "finish_reason": None}],
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


async def _stream_chat_generate(req_prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int | None):
    """Like _stream_generate but yields OpenAI chat format SSE chunks."""
    token_ids = encode(req_prompt)
    seq_len = token_ids.shape[1]

    if seed is not None:
        np.random.seed(seed)

    queue: asyncio.Queue = asyncio.Queue(maxsize=0)
    loop = asyncio.get_running_loop()

    def _run():
        try:
            for token_id, step_time in model.generate_stream_gpu(
                token_ids, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
                eos_token_id=tokenizer.eos_token_id,
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
    all_ids: list[int] = []
    prev_text = ""
    while True:
        msg_type, payload, step_time = await queue.get()
        if msg_type == "token":
            all_ids.append(payload)
            # Incremental decode: accumulate IDs for proper whitespace
            full = decode(all_ids)
            if full.startswith(prev_text):
                delta = full[len(prev_text):]
            else:
                delta = full
            prev_text = full
            if not delta:
                step_times.append(step_time)
                continue
            chunk = {
                "choices": [{"delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            step_times.append(step_time)
        elif msg_type == "done":
            break
        elif msg_type == "error":
            yield f"data: {json.dumps({'error': payload})}\n\n"
            return

    num_decodes = len(step_times)
    total_time = sum(step_times)
    finish = "stop" if num_decodes > 0 else "length"
    final_chunk = {
        "choices": [{"delta": {}, "finish_reason": finish}],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    ttft = step_times[0] if step_times else 0.0
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
        _validate_request(seq_len, req.max_tokens, max_context_positions(model))
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
    if is_gpt2_variant(model_variant):
        raise HTTPException(400, "Chat completions are not supported for GPT-2; use /v1/completions")
    if req.stream:
        prompt = format_chat_prompt([m.model_dump() for m in req.messages])
        token_ids = encode(prompt)
        seq_len = token_ids.shape[1]
        _validate_request(seq_len, req.max_tokens, max_context_positions(model))
        return StreamingResponse(
            _stream_chat_generate(prompt, req.max_tokens, req.temperature, req.top_k, req.seed),
            media_type="text/event-stream",
        )
    prompt = format_chat_prompt([m.model_dump() for m in req.messages])
    new_text, new_ids, seq_len, dt = _generate(
        prompt, req.max_tokens, req.temperature, req.top_k, req.seed
    )
    finish = "stop" if new_ids and new_ids[-1] == tokenizer.eos_token_id else "length"
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
                finish_reason=finish,
            )
        ],
    )


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Triton HTTP server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--model", type=str, default="SmolLM2-135M",
                   choices=[
                       "SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B",
                       "SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct", "SmolLM2-1.7B-Instruct",
                       "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                   ])
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
