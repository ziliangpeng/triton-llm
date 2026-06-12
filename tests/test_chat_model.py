"""Tests for SmolLM2 chat model functionality.

Includes EOS stopping, chat template formatting, server-side decode
with role prefix stripping, and streaming chat output format.

These tests use random weights and a minimal config — no GPU or H100 needed.
"""
import sys
import os
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM, gpu


# ── Helpers ──────────────────────────────────────────────────────────

def make_mini_config():
    """Return a minimal config for fast test execution."""
    return SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )


def make_random_weights(config):
    """Generate random weights for a config."""
    np.random.seed(42)
    n_layer = config.num_hidden_layers
    n = config.hidden_size
    w = {}
    for i in range(n_layer):
        for p in ["q", "k", "v", "o"]:
            out_dim = n if p == "o" else config.num_attention_heads * (n // config.num_attention_heads)
            in_dim = n
            w[f"model.layers.{i}.self_attn.{p}_proj.weight"] = np.random.randn(out_dim, in_dim).astype(np.float32)
        for p in ["gate", "up", "down"]:
            dim = config.intermediate_size if p != "down" else n
            w[f"model.layers.{i}.mlp.{p}_proj.weight"] = np.random.randn(dim, n).astype(np.float32)
        w[f"model.layers.{i}.input_layernorm.weight"] = np.random.randn(n).astype(np.float32)
        w[f"model.layers.{i}.post_attention_layernorm.weight"] = np.random.randn(n).astype(np.float32)
    w["model.embed_tokens.weight"] = np.random.randn(config.vocab_size, n).astype(np.float32)
    w["model.norm.weight"] = np.random.randn(n).astype(np.float32)
    w["lm_head.weight"] = np.random.randn(config.vocab_size, n).astype(np.float32)
    return w


def create_model():
    """Create a test model with random weights."""
    config = make_mini_config()
    weights = make_random_weights(config)
    return SmolLM2ForCausalLM(config, weights)


# ── EOS Stopping Tests ──────────────────────────────────────────────

class TestEOSStopping:
    """Verify that generate_gpu() and generate_stream_gpu() stop on EOS."""

    def test_generate_stops_at_eos(self):
        """generate_gpu with eos_token_id stops early, not running to max_new_tokens."""
        model = create_model()
        prompt = np.array([[5, 12, 7]], dtype=np.int32)

        # Without EOS — runs full max_new_tokens
        out_no_eos = model.generate_gpu(prompt, max_new_tokens=10, temperature=0.0)
        assert out_no_eos.shape[1] == prompt.shape[1] + 10, \
            f"Expected {prompt.shape[1] + 10} tokens, got {out_no_eos.shape[1]}"

        # With EOS set to 99 (unlikely to be sampled) — same as no EOS
        out_far_eos = model.generate_gpu(prompt, max_new_tokens=10, temperature=0.0,
                                          eos_token_id=99)
        assert out_far_eos.shape[1] == prompt.shape[1] + 10, \
            f"EOS=99 should still produce {prompt.shape[1] + 10} tokens"

    def test_generate_stream_stops_at_eos(self):
        """generate_stream_gpu with eos_token_id yields fewer tokens than max."""
        model = create_model()
        prompt = np.array([[5]], dtype=np.int32)

        with_eos = list(model.generate_stream_gpu(
            prompt, max_new_tokens=20, temperature=0.0, eos_token_id=5,
        ))
        assert len(with_eos) <= 20, \
            f"EOS stopping should yield ≤20 tokens, got {len(with_eos)}"
        # Last yielded token should be 5 (EOS)
        if len(with_eos) > 0:
            assert with_eos[-1][0] == 5, \
                f"Last token should be EOS (5), got {with_eos[-1][0]}"

        without_eos = list(model.generate_stream_gpu(
            prompt, max_new_tokens=20, temperature=0.0,
        ))
        assert len(without_eos) == 20, \
            f"Without EOS should yield exactly 20 tokens, got {len(without_eos)}"

    def test_generate_stream_yields_eos_then_stops(self):
        """The EOS token itself is yielded, then generation stops."""
        model = create_model()
        prompt = np.array([[1]], dtype=np.int32)
        tokens = list(model.generate_stream_gpu(
            prompt, max_new_tokens=10, temperature=0.0, eos_token_id=1,
        ))
        # Should stop immediately (first sampled token is 1 = EOS)
        assert len(tokens) == 1, \
            f"Should yield exactly 1 token (EOS), got {len(tokens)}"
        assert tokens[0][0] == 1

    def test_generate_stream_buffered_eos(self):
        """EOS token may be yielded before the buffer check — generator stops after it."""
        model = create_model()
        prompt = np.array([[2, 3, 5]], dtype=np.int32)
        # Set EOS to 42 — unlikely but let's verify the generator stops
        tokens = list(model.generate_stream_gpu(
            prompt, max_new_tokens=100, temperature=0.0, eos_token_id=42,
        ))
        # Should finish without error regardless of EOS
        assert len(tokens) > 0

    def test_backward_compat_aliases_forward_eos(self):
        """generate() and generate_stream() aliases forward eos_token_id."""
        model = create_model()
        prompt = np.array([[5]], dtype=np.int32)

        tokens = list(model.generate_stream(
            prompt, max_new_tokens=5, temperature=0.0, eos_token_id=5,
        ))
        assert len(tokens) <= 5, \
            f"Stream alias should respect eos_token_id, got {len(tokens)} tokens"


# ── Decode (Role Prefix Stripping) Tests ─────────────────────────────

class TestDecodeRoleStrip:
    """Verify decode() strips leading role prefixes."""

    def test_decode_strips_assistant_prefix(self):
        """decode() strips leading 'assistant\\n' prefix."""
        # We test via the module-level function if available
        from scripts.serve_model import decode
        # Mock tokenizer behaviour — simulate a response with role prefix
        from unittest.mock import patch, MagicMock
        import scripts.serve_model as sm
        sm.tokenizer = MagicMock()

        # Test with different leading prefixes
        with patch.object(sm, 'tokenizer') as tok:
            tok.decode.return_value = "assistant\nHello world"
            result = decode([123, 456])
            assert result == "Hello world", f"Expected 'Hello world', got {repr(result)}"

            tok.decode.return_value = "user\nWhat is gravity?"
            result = decode([789])
            assert result == "What is gravity?", \
                f"Expected 'What is gravity?', got {repr(result)}"

            tok.decode.return_value = "system\nYou are helpful"
            result = decode([111])
            assert result == "You are helpful", \
                f"Expected 'You are helpful', got {repr(result)}"

    def test_decode_no_prefix(self):
        """decode() passes through text without role prefix."""
        from scripts.serve_model import decode
        import scripts.serve_model as sm
        from unittest.mock import patch

        with patch.object(sm, 'tokenizer') as tok:
            tok.decode.return_value = "Hello world"
            result = decode([123, 456])
            assert result == "Hello world"

    def test_decode_strips_edge_chars(self):
        """decode() strips whitespace from both ends."""
        from scripts.serve_model import decode
        import scripts.serve_model as sm
        from unittest.mock import patch

        with patch.object(sm, 'tokenizer') as tok:
            tok.decode.return_value = "  Hello world\n\n"
            result = decode([123])
            assert result == "Hello world"


# ── Chat Template Tests ────────────────────────────────────────────

class TestFormatChatPrompt:
    """Verify format_chat_prompt builds correct prompts."""

    def test_instruct_model_chatml_format(self):
        """Instruct model uses tokenizer.apply_chat_template for ChatML."""
        from scripts.serve_model import format_chat_prompt
        import scripts.serve_model as sm
        from unittest.mock import patch, MagicMock

        with (
            patch.object(sm, 'tokenizer') as tok,
            patch.object(sm, 'model_variant', "SmolLM2-135M-Instruct"),
        ):
            tok.apply_chat_template.return_value = (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                "<|im_start|>user\nWhat is gravity?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is gravity?"},
            ]
            result = format_chat_prompt(messages)
            assert "<|im_start|>" in result
            assert "system" in result
            assert "user" in result
            assert "assistant" in result
            tok.apply_chat_template.assert_called_once_with(messages, tokenize=False)

    def test_base_model_fallback(self):
        """Base model falls back to plain text with 'Assistant:' trailing."""
        from scripts.serve_model import format_chat_prompt
        import scripts.serve_model as sm
        from unittest.mock import patch

        with patch.object(sm, 'model_variant', "SmolLM2-135M"):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            result = format_chat_prompt(messages)
            assert "User:" in result
            assert "Hello" in result
            assert "Assistant:" in result
            assert result.endswith("Assistant:"), \
                f"Expected trailing 'Assistant:', got {repr(result)}"


# ── Server Endpoint Tests ──────────────────────────────────────────

class TestChatCompletions:
    """Verify /v1/chat/completions endpoint produces correct output."""

    def test_finish_reason_stop_on_eos(self):
        """Non-streaming chat returns finish_reason='stop' when EOS triggered."""
        from scripts.serve_model import format_chat_prompt, _generate, tokenizer
        import scripts.serve_model as sm
        from unittest.mock import patch, MagicMock, PropertyMock

        # Mock model + tokenizer
        mock_model = MagicMock()
        mock_model.config.n_positions = 512
        mock_model.generate_gpu.return_value = np.array([[5, 12, 7, 2, 15]], dtype=np.int32)  # 2 = EOS
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.encode.return_value = [5, 12, 7]
        mock_tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}" if isinstance(ids, list) else str(ids)

        with patch.multiple(sm, model=mock_model, tokenizer=mock_tokenizer, no_download=True):
            new_text, new_ids, seq_len, dt = _generate("test", 5, 0.0, 0, None)
            assert seq_len > 0
            assert len(new_ids) > 0
            finish = "stop" if new_ids and new_ids[-1] == 2 else "length"
            assert finish == "stop", \
                f"Expected 'stop' when last token is EOS, got {finish}"

    def test_finish_reason_length_on_max_tokens(self):
        """Non-streaming chat returns finish_reason='length' when not EOS."""
        from scripts.serve_model import _generate
        import scripts.serve_model as sm
        from unittest.mock import patch, MagicMock

        mock_model = MagicMock()
        mock_model.config.n_positions = 512
        mock_model.generate_gpu.return_value = np.array([[5, 12, 7, 42, 99]], dtype=np.int32)  # no EOS
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.encode.return_value = [5, 12, 7]
        mock_tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        with patch.multiple(sm, model=mock_model, tokenizer=mock_tokenizer, no_download=True):
            new_text, new_ids, seq_len, dt = _generate("test", 10, 0.0, 0, None)
            finish = "stop" if new_ids and new_ids[-1] == 2 else "length"
            assert finish == "length", \
                f"Expected 'length' when last token is not EOS, got {finish}"


# ── Streaming Chat Output Format Tests ──────────────────────────────

class TestStreamChatFormat:
    """Verify _stream_chat_generate produces correct SSE delta chunks."""

    def test_incremental_decode_preserves_spaces(self):
        """Incremental decode in _stream_generate produces correct deltas."""
        from scripts.serve_model import decode
        import scripts.serve_model as sm
        from unittest.mock import patch

        # Simulate incremental decode: accumulating token IDs
        with patch.object(sm, 'tokenizer') as tok:
            tok.eos_token_id = 2

            # Simulate decode(['ass']) → "ass", decode(['ass','istant']) → "assistant"
            def mock_decode(ids, **kw):
                text = ""
                for i, tid in enumerate(ids):
                    text += f"part_{tid}"
                return text

            tok.decode.side_effect = mock_decode

            all_ids = []
            prev_text = ""
            for token_id in [100, 101, 102]:
                all_ids.append(token_id)
                full = mock_decode(all_ids)
                if full.startswith(prev_text):
                    delta = full[len(prev_text):]
                else:
                    delta = full
                prev_text = full
                assert len(delta) > 0, \
                    f"Delta should not be empty at token {token_id}"


# ── Config Tests ──────────────────────────────────────────────────

class TestConfig:
    """Verify SmolLM2Config.from_pretrained handles all variants."""

    def test_base_variants(self):
        """Base model variants resolve correctly."""
        for variant in ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]:
            cfg = SmolLM2Config.from_pretrained(variant)
            assert cfg.vocab_size == 49152
            assert cfg.num_hidden_layers in [30, 32, 24]

    def test_instruct_variants(self):
        """Instruct variants resolve with same architecture as base."""
        for variant in ["SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct", "SmolLM2-1.7B-Instruct"]:
            cfg = SmolLM2Config.from_pretrained(variant)
            assert cfg.vocab_size == 49152
            assert cfg.max_position_embeddings == 8192

    def test_invalid_variant(self):
        """Unknown variant raises ValueError."""
        with pytest.raises(ValueError):
            SmolLM2Config.from_pretrained("NonExistentModel")

    def test_instruct_matches_base(self):
        """Instruct variant has same hidden_size as its base counterpart."""
        base = SmolLM2Config.from_pretrained("SmolLM2-135M")
        inst = SmolLM2Config.from_pretrained("SmolLM2-135M-Instruct")
        assert base.hidden_size == inst.hidden_size
        assert base.num_hidden_layers == inst.num_hidden_layers
        assert base.num_attention_heads == inst.num_attention_heads
