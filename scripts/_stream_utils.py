"""Internal SSE streaming helpers — shared by client.py and repl.py."""

import json


def parse_sse_lines(resp):
    """Read SSE lines from an HTTP response, yield (text, is_last, usage).

    Yields triples:
      * (token_text, False, None) — a generated token
      * ("", True, None) — final finish_reason chunk, no text
      * ("", True, {"prompt_tokens": N, "completion_tokens": N, ...}) — usage
        stats chunk (emitted by the server before [DONE] when
        ``stream_options.include_usage`` is set).

    Stops on ``data: [DONE]`` or end-of-stream.
    """
    for line in resp:
        line = line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
            usage = chunk.get("usage")
            choices = chunk.get("choices", [])
            if usage is not None:
                # Usage stats chunk (choices is empty array)
                yield "", True, usage
            elif choices:
                text = choices[0].get("text", "")
                is_last = choices[0].get("finish_reason") is not None
                yield text, is_last, None
        except json.JSONDecodeError:
            pass
