"""Contract tests for Llama-3.2-3B-Instruct bring-up.

These stay at the config / serving layer so they can run without Triton or GPU.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.responses import StreamingResponse

import scripts.serve_model as sm
from scripts.serve_model import ChatCompletionRequest, ChatMessage, ChatCompletionResponse

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "smollm2_triton" / "config.py"
_spec = importlib.util.spec_from_file_location("llama_cfg", _CONFIG_PATH)
_cfg_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_cfg_mod)
SmolLM2Config = _cfg_mod.SmolLM2Config

VARIANT = "Llama-3.2-3B-Instruct"


def _mk_mock_model():
    model = MagicMock()
    # Llama-style config; set both names so helper works even on MagicMock.
    model.config.n_positions = 131072
    model.config.max_position_embeddings = 131072
    model.generate_gpu.return_value = sm.np.array([[30, 31, 99]], dtype=sm.np.int32)
    model.generate_stream_gpu.return_value = iter([(1, 0.10), (2, 0.11), (99, 0.12)])
    return model


def _mk_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 99
    tok.encode.return_value = [30, 31]

    def _apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        base = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        if add_generation_prompt:
            return base + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return base

    tok.apply_chat_template.side_effect = _apply_chat_template

    def _decode(ids, skip_special_tokens=True):
        mapping = {
            (99,): "assistant\nHello",
            (30, 31, 99): "assistant\nHello",
            (1,): "ass",
            (1, 2): "assistant",
            (1, 2, 99): "assistant\nHello",
        }
        return mapping.get(tuple(ids), "Hello")

    tok.decode.side_effect = _decode
    return tok


async def _collect_streaming_payload(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(parts)


def _parse_sse(payload: str):
    events = []
    for block in payload.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        data = block[6:]
        if data == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(sm.json.loads(data))
    return events


def test_llama32_config_variant_exists():
    cfg = SmolLM2Config.from_pretrained(VARIANT)
    assert cfg.hidden_size == 3072
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 24
    assert cfg.num_key_value_heads == 8
    assert cfg.intermediate_size == 8192
    assert cfg.vocab_size == 128256
    assert cfg.repo_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_hf_repo_id_for_llama32():
    assert sm.hf_repo_id(VARIANT) == "meta-llama/Llama-3.2-3B-Instruct"


def test_parse_args_accepts_llama32():
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["serve_model.py", "--model", VARIANT]
        args = sm.parse_args()
    finally:
        sys.argv = old
    assert args.model == VARIANT


def test_format_chat_prompt_uses_tokenizer_template_for_llama32():
    tok = _mk_mock_tokenizer()
    messages = [{"role": "user", "content": "Hi"}]
    with patch.object(sm, "model_variant", VARIANT), patch.object(sm, "tokenizer", tok):
        prompt = sm.format_chat_prompt(messages)
    assert "assistant" in prompt
    tok.apply_chat_template.assert_called_once_with(messages, tokenize=False, add_generation_prompt=True)


def test_chat_endpoint_non_streaming_for_llama32():
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=5, temperature=0.0)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", VARIANT), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.chat_completions(req))
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello"
    assert resp.choices[0].finish_reason == "stop"


def test_chat_endpoint_streaming_for_llama32():
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=5, temperature=0.0, stream=True)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", VARIANT), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.chat_completions(req))
        assert isinstance(resp, StreamingResponse)
        payload = asyncio.run(_collect_streaming_payload(resp))
    events = _parse_sse(payload)
    choice_events = [e for e in events if isinstance(e, dict) and e.get("choices")]
    text = "".join(e["choices"][0].get("delta", {}).get("content", "") for e in choice_events)
    assert text.endswith("Hello")
    assert any(e["choices"][0].get("finish_reason") == "stop" for e in choice_events)
    assert events[-1] == "[DONE]"


def test_download_weights_supports_sharded_safetensors_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "model.safetensors.index.json"
        index_path.write_text(json.dumps({
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
            }
        }))
        shard1 = str(Path(tmpdir) / "model-00001-of-00002.safetensors")
        shard2 = str(Path(tmpdir) / "model-00002-of-00002.safetensors")

        def fake_download(repo_id, filename, cache_dir):
            if filename == "model.safetensors.index.json":
                return str(index_path)
            if filename == "model-00001-of-00002.safetensors":
                return shard1
            if filename == "model-00002-of-00002.safetensors":
                return shard2
            raise AssertionError(filename)

        def fake_load_file(path, device="cpu"):
            if path == shard1:
                import torch
                return {"a.weight": torch.ones((2, 2))}
            if path == shard2:
                import torch
                return {"b.weight": torch.zeros((2, 2))}
            raise AssertionError(path)

        with patch("tempfile.gettempdir", return_value=tmpdir), \
             patch("huggingface_hub.list_repo_files", return_value=[
                 "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors",
                 "model-00002-of-00002.safetensors",
             ]), \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_download), \
             patch("safetensors.torch.load_file", side_effect=fake_load_file):
            weights = sm.download_weights(VARIANT)

        assert "a.weight" in weights and "b.weight" in weights
        assert weights["a.weight"].shape == (2, 2)
        assert weights["b.weight"].shape == (2, 2)
