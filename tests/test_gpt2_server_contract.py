"""Server-contract tests for GPT-2 restoration.

These tests avoid importing the real GPT-2 model implementation, so they can
run without Triton/GPU locally. They verify only the serving-layer contracts.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import scripts.serve_model as sm
from scripts.serve_model import ChatCompletionRequest, ChatMessage


@pytest.mark.parametrize("variant", ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
def test_is_gpt2_variant_true(variant):
    assert sm.is_gpt2_variant(variant) is True


@pytest.mark.parametrize("variant", ["SmolLM2-135M", "SmolLM2-135M-Instruct"])
def test_is_gpt2_variant_false(variant):
    assert sm.is_gpt2_variant(variant) is False


@pytest.mark.parametrize("variant", ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
def test_hf_repo_id_for_gpt2_variants(variant):
    assert sm.hf_repo_id(variant) == f"openai-community/{variant}"


def test_chat_endpoint_rejects_gpt2():
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    with patch.object(sm, "model_variant", "gpt2"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(sm.chat_completions(req))
    assert exc.value.status_code == 400
    assert "not supported for GPT-2" in str(exc.value.detail)


def test_max_context_positions_supports_gpt2_and_smollm2_shapes():
    class GPT2Cfg:
        n_positions = 1024

    class SmolCfg:
        max_position_embeddings = 8192

    class M1:
        config = GPT2Cfg()

    class M2:
        config = SmolCfg()

    assert sm.max_context_positions(M1()) == 1024
    assert sm.max_context_positions(M2()) == 8192


def test_decode_preserves_gpt2_completion_whitespace():
    tok = type("Tok", (), {"decode": lambda self, ids, skip_special_tokens=True: " hello\n"})()
    with patch.object(sm, "tokenizer", tok):
        assert sm.decode([1, 2]) == " hello\n"


def test_decode_can_strip_chat_role_prefix_when_requested():
    tok = type("Tok", (), {"decode": lambda self, ids, skip_special_tokens=True: "assistant\nHello"})()
    with patch.object(sm, "tokenizer", tok):
        assert sm.decode([1], strip_role_prefix=True, strip_whitespace=True) == "Hello"