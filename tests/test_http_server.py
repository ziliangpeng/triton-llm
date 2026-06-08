"""Tests for HTTP server schemas, utilities, and client.

These tests verify Pydantic validation, prompt formatting, and
client-side helpers — no GPU or model required.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add repo root to path so we can import server modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Schema validation (TestClient-free) ──────────────────────────────

class TestCompletionSchema:
    """Test /v1/completions request/response validation."""

    def test_valid_request(self):
        from scripts.serve_model import CompletionRequest
        req = CompletionRequest(prompt="Hello", max_tokens=10, temperature=0.5)
        assert req.prompt == "Hello"
        assert req.max_tokens == 10
        assert req.temperature == 0.5
        assert req.top_k == 0
        assert req.seed is None

    def test_valid_request_with_seed(self):
        from scripts.serve_model import CompletionRequest
        req = CompletionRequest(prompt="Test", seed=42)
        assert req.seed == 42

    def test_empty_prompt_rejected(self):
        from scripts.serve_model import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="")  # min_length=1 on Field rejects empty strings

    def test_max_tokens_bounds(self):
        from scripts.serve_model import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="x", max_tokens=0)
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="x", max_tokens=99999)

    def test_temperature_bounds(self):
        from scripts.serve_model import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="x", temperature=-0.5)
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="x", temperature=3.0)

    def test_completion_response_roundtrip(self):
        from scripts.serve_model import CompletionResponse, Usage
        resp = CompletionResponse(
            text="hello world",
            tokens=[1, 2, 3],
            usage=Usage(prompt_tokens=1, completion_tokens=3, total_tokens=4, time_seconds=0.5),
        )
        d = json.loads(resp.model_dump_json())
        assert d["text"] == "hello world"
        assert d["tokens"] == [1, 2, 3]
        assert d["usage"]["prompt_tokens"] == 1
        assert d["usage"]["total_tokens"] == 4


class TestChatSchema:
    """Test /v1/chat/completions request/response validation."""

    def test_valid_request(self):
        from scripts.serve_model import ChatCompletionRequest, ChatMessage
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=20,
        )
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content == "Hi"

    def test_min_messages(self):
        from scripts.serve_model import ChatCompletionRequest
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[])

    def test_multiple_messages(self):
        from scripts.serve_model import ChatCompletionRequest, ChatMessage
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        assert len(req.messages) == 2

    def test_chat_response_roundtrip(self):
        from scripts.serve_model import (
            ChatCompletionResponse,
            ChatResponseChoice,
            ChatMessage,
            Usage,
        )
        resp = ChatCompletionResponse(
            choices=[
                ChatResponseChoice(
                    message=ChatMessage(role="assistant", content="Hi there"),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7, time_seconds=0.3),
        )
        d = json.loads(resp.model_dump_json())
        assert d["choices"][0]["message"]["content"] == "Hi there"
        assert d["usage"]["prompt_tokens"] == 5


# ── format_chat_prompt ───────────────────────────────────────────────

class TestFormatChatPrompt:
    """Test the plain-text chat prompt formatter."""

    def test_single_user_message(self):
        from scripts.serve_model import format_chat_prompt
        result = format_chat_prompt([{"role": "user", "content": "Hello"}])
        assert result == "User: Hello"

    def test_system_and_user(self):
        from scripts.serve_model import format_chat_prompt
        result = format_chat_prompt([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ])
        assert "System: You are helpful." in result
        assert "User: What is 2+2?" in result

    def test_multi_turn(self):
        from scripts.serve_model import format_chat_prompt
        result = format_chat_prompt([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ])
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "User: Hi"
        assert lines[1] == "Assistant: Hello!"
        assert lines[2] == "User: How are you?"

    def test_default_role(self):
        from scripts.serve_model import format_chat_prompt
        # Missing role defaults to "User"
        result = format_chat_prompt([{"content": "test"}])
        assert result == "User: test"

    def test_empty_message_list(self):
        from scripts.serve_model import format_chat_prompt
        result = format_chat_prompt([])
        assert result == ""


# ── Client query functions ───────────────────────────────────────────

class TestClientQueries:
    """Test client query functions with a real mock HTTP server."""

    @staticmethod
    def _unused_port() -> int:
        """Return a port that is guaranteed to be unused at this instant."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_query_completions_connection_refused(self):
        """Should return error dict when no server is listening."""
        from scripts.client import query_completions
        port = self._unused_port()
        result = query_completions("test", host="127.0.0.1", port=port)
        assert "error" in result
        assert "Connection failed" in result["error"] or "refused" in result["error"]

    def test_query_chat_connection_refused(self):
        """Should return error dict when no server is listening."""
        from scripts.client import query_chat
        port = self._unused_port()
        result = query_chat([{"role": "user", "content": "test"}],
                            host="127.0.0.1", port=port)
        assert "error" in result
        assert "Connection failed" in result["error"] or "refused" in result["error"]
