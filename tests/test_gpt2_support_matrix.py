"""Comprehensive support-matrix / endpoint / sanity tests for GPT-2 variants.

These tests stay at the server/contract layer, so they can run without Triton
or a GPU locally while still exercising the GPT-2 serving surface.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import scripts.serve_model as sm
from scripts.serve_model import ChatCompletionRequest, ChatMessage, CompletionRequest, CompletionResponse


GPT2_VARIANTS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def _mk_mock_model():
    model = MagicMock()
    # GPT-2 config uses n_positions
    model.config.n_positions = 1024
    model.generate_gpu.return_value = sm.np.array([[42, 43, 99]], dtype=sm.np.int32)
    model.generate_stream_gpu.return_value = iter([(1, 0.10), (2, 0.11), (99, 0.12)])
    return model


def _mk_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 99
    tok.encode.side_effect = lambda text: [42, 43] if text == "Hello" else [10, 11]

    def _decode(ids, skip_special_tokens=True):
        mapping = {
            (99,): " hello",              # preserve GPT-2 leading whitespace
            (42, 43, 99): " hello",       # non-streaming completion output
            (1,): " h",
            (1, 2): " he",
            (1, 2, 99): " hello",
        }
        return mapping.get(tuple(ids), " hello")

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


@patch.object(sm, "tokenizer", _mk_mock_tokenizer())
def test_decode_preserves_leading_whitespace_for_gpt2_completions():
    assert sm.decode([42, 43, 99]) == " hello"


@patch.object(sm, "tokenizer", _mk_mock_tokenizer())
def test_decode_can_still_strip_chat_prefix_when_requested():
    sm.tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "assistant\nHello"
    assert sm.decode([1], strip_role_prefix=True, strip_whitespace=True) == "Hello"


def test_parse_args_accepts_all_gpt2_variants():
    import sys

    for variant in GPT2_VARIANTS:
        old = sys.argv[:]
        try:
            sys.argv = ["serve_model.py", "--model", variant]
            args = sm.parse_args()
        finally:
            sys.argv = old
        assert args.model == variant


@patch.object(sm, "model", _mk_mock_model())
@patch.object(sm, "tokenizer", _mk_mock_tokenizer())
def test_completion_endpoint_supports_all_gpt2_variants():
    req = CompletionRequest(prompt="Hello", max_tokens=5, temperature=0.0)
    for variant in GPT2_VARIANTS:
        with patch.object(sm, "model_variant", variant):
            resp = asyncio.run(sm.completions(req))
        assert isinstance(resp, CompletionResponse)
        assert resp.text == " hello"
        assert resp.tokens == [99]
        assert resp.usage.prompt_tokens == 2
        assert resp.usage.completion_tokens == 1


def test_streaming_completion_supports_all_gpt2_variants_and_preserves_text():
    req = CompletionRequest(prompt="Hello", max_tokens=5, temperature=0.0, stream=True)
    for variant in GPT2_VARIANTS:
        model = _mk_mock_model()
        tok = _mk_mock_tokenizer()
        with patch.object(sm, "model", model), patch.object(sm, "tokenizer", tok), patch.object(sm, "model_variant", variant):
            resp = asyncio.run(sm.completions(req))
            assert isinstance(resp, StreamingResponse)
            payload = asyncio.run(_collect_streaming_payload(resp))
        events = _parse_sse(payload)
        choice_events = [e for e in events if isinstance(e, dict) and e.get("choices")]
        usage_events = [e for e in events if isinstance(e, dict) and e.get("usage")]
        text = "".join(e["choices"][0].get("text", "") for e in choice_events)
        assert text == " hello"
        assert any(e["choices"][0].get("finish_reason") == "stop" for e in choice_events)
        assert len(usage_events) == 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 5
        assert "ttft_ms" in usage and "tokens_per_second" in usage
        assert events[-1] == "[DONE]"


@patch.object(sm, "model", _mk_mock_model())
@patch.object(sm, "tokenizer", _mk_mock_tokenizer())
@pytest.mark.parametrize("variant", GPT2_VARIANTS)
def test_chat_endpoint_rejects_gpt2_with_clear_error(variant):
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    with patch.object(sm, "model_variant", variant):
        try:
            asyncio.run(sm.chat_completions(req))
            raise AssertionError("Expected GPT-2 chat request to be rejected")
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "not supported for GPT-2" in str(exc.detail)


@patch.object(sm, "tokenizer", _mk_mock_tokenizer())
def test_completion_endpoint_tolerates_oversized_top_k_via_model_sampling_contract():
    req = CompletionRequest(prompt="Hello", max_tokens=5, temperature=1.0, top_k=999999)
    model = _mk_mock_model()
    with patch.object(sm, "model", model), patch.object(sm, "model_variant", "gpt2"):
        resp = asyncio.run(sm.completions(req))
    assert isinstance(resp, CompletionResponse)
    assert resp.text == " hello"
    assert model.generate_gpu.call_args.kwargs["top_k"] == 999999
    assert model.generate_gpu.call_args.kwargs["eos_token_id"] == 99


def test_max_context_positions_uses_gpt2_n_positions():
    class Cfg:
        n_positions = 1024

    class Model:
        config = Cfg()

    assert sm.max_context_positions(Model()) == 1024
