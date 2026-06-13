"""Support-matrix / endpoint-contract tests for SmolLM2 variants.

Goal: codify which model variants are supported today, and verify the
server endpoints behave correctly for base vs instruct variants without
needing real model weights or a GPU.

GPT-2 is intentionally excluded for now; restoration is tracked by issue #39.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util

import pytest
from fastapi.responses import StreamingResponse

import scripts.serve_model as sm
from scripts.serve_model import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "smollm2_triton" / "config.py"
_spec = importlib.util.spec_from_file_location("smollm2_config", _CONFIG_PATH)
_cfg_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_cfg_mod)
SmolLM2Config = _cfg_mod.SmolLM2Config


SUPPORTED_VARIANTS = [
    "SmolLM2-135M",
    "SmolLM2-360M",
    "SmolLM2-1.7B",
    "SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-Instruct",
]

BASE_VARIANTS = [v for v in SUPPORTED_VARIANTS if "Instruct" not in v]
INSTRUCT_VARIANTS = [v for v in SUPPORTED_VARIANTS if "Instruct" in v]


def _mk_mock_model():
    model = MagicMock()
    model.config.n_positions = 8192
    # prompt len assumed 2; last token = eos => finish_reason stop
    model.generate_gpu.return_value = sm.np.array([[10, 11, 99]], dtype=sm.np.int32)
    model.generate_stream_gpu.return_value = iter([(1, 0.10), (2, 0.11), (99, 0.12)])
    return model


def _mk_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 99

    def _encode(text):
        # Distinguish plain completion prompts from formatted chat prompts.
        if text == "Hi":
            return [10, 11]
        if text.endswith("\nAssistant:"):
            return [20, 21]
        if "<|im_start|>assistant" in text:
            return [30, 31]
        return [10, 11]

    tok.encode.side_effect = _encode

    def _decode(ids, skip_special_tokens=True):
        # Simulate prefix stripping path for non-streaming and good spacing for streaming.
        if ids == [99]:
            return "assistant\nHello"
        mapping = {
            (1,): "ass",
            (1, 2): "assistant",
            (1, 2, 99): "assistant Hello",
            (10, 11, 99): "assistant\nHello",
        }
        return mapping.get(tuple(ids), "Hello")

    tok.decode.side_effect = _decode
    tok.apply_chat_template.return_value = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tok


async def _collect_streaming_text(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            parts.append(chunk.decode())
        else:
            parts.append(chunk)
    return "".join(parts)


def _parse_sse_payload(payload: str):
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


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_config_support_matrix_has_all_smollm2_variants(variant):
    cfg = SmolLM2Config.from_pretrained(variant)
    assert cfg.vocab_size == 49152
    assert cfg.max_position_embeddings == 8192


@pytest.mark.parametrize("variant", BASE_VARIANTS)
def test_base_variants_use_plaintext_chat_fallback(variant):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    with patch.object(sm, "model_variant", variant):
        result = sm.format_chat_prompt(messages)
    assert result.endswith("\nAssistant:")
    assert "User: Hi" in result
    assert "Assistant: Hello" in result


@pytest.mark.parametrize("variant", INSTRUCT_VARIANTS)
def test_instruct_variants_use_chat_template(variant):
    messages = [{"role": "user", "content": "Hi"}]
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "tokenizer", tok):
        result = sm.format_chat_prompt(messages)
    assert "<|im_start|>assistant" in result
    tok.apply_chat_template.assert_called_once_with(messages, tokenize=False)


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_completion_endpoint_supported_for_all_smollm2_variants(variant):
    req = CompletionRequest(prompt="Hi", max_tokens=5, temperature=0.0)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.completions(req))
    assert isinstance(resp, CompletionResponse)
    assert resp.text == "Hello"
    assert resp.usage.prompt_tokens == 2
    assert resp.usage.completion_tokens == 1


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_chat_endpoint_supported_for_all_smollm2_variants(variant):
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=5, temperature=0.0)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.chat_completions(req))
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello"
    assert resp.choices[0].finish_reason == "stop"
    # Verify the route encoded the formatted prompt path appropriate for the variant.
    called_ids = model.generate_gpu.call_args.args[0].tolist()[0]
    if "Instruct" in variant:
        tok.apply_chat_template.assert_called_once()
        assert called_ids == [30, 31]
    else:
        tok.apply_chat_template.assert_not_called()
        assert called_ids == [20, 21]


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_streaming_completion_supported_for_all_smollm2_variants(variant):
    req = CompletionRequest(prompt="Hi", max_tokens=5, temperature=0.0, stream=True)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.completions(req))
        assert isinstance(resp, StreamingResponse)
        payload = asyncio.run(_collect_streaming_text(resp))
    events = _parse_sse_payload(payload)
    choice_events = [e for e in events if isinstance(e, dict) and e.get("choices")]
    text = "".join(e["choices"][0].get("text", "") for e in choice_events)
    assert text.endswith("Hello")
    assert any(e["choices"][0].get("finish_reason") in ("length", "stop") for e in choice_events)
    assert events[-1] == "[DONE]"


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_streaming_chat_supported_for_all_smollm2_variants(variant):
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=5, temperature=0.0, stream=True)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.chat_completions(req))
        assert isinstance(resp, StreamingResponse)
        payload = asyncio.run(_collect_streaming_text(resp))
    events = _parse_sse_payload(payload)
    choice_events = [e for e in events if isinstance(e, dict) and e.get("choices")]
    text = "".join(
        e["choices"][0].get("delta", {}).get("content", "")
        for e in choice_events
    )
    # Today the server may still emit assistant-prefixed streaming text; the
    # client-side prefix-strip helper is tested separately. Here we only assert
    # the endpoint contract and that content reconstructs successfully.
    assert text.endswith("Hello")
    assert any(e["choices"][0].get("delta", {}).get("content", "") for e in choice_events)
    assert any(e["choices"][0].get("finish_reason") == "stop" for e in choice_events)
    assert events[-1] == "[DONE]"


@pytest.mark.parametrize("variant", INSTRUCT_VARIANTS)
def test_chat_quality_sanity_no_role_prefix_leak_non_streaming(variant):
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")], max_tokens=5, temperature=0.0)
    model = _mk_mock_model()
    tok = _mk_mock_tokenizer()
    with patch.object(sm, "model_variant", variant), patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok):
        resp = asyncio.run(sm.chat_completions(req))
    text = resp.choices[0].message.content
    assert not text.startswith("assistant")
    assert not text.startswith("user")
    assert not text.startswith("system")


def test_gpt2_support_matrix_pending_restore():
    pytest.skip("GPT-2 support is intentionally pending restore in issue #39")
