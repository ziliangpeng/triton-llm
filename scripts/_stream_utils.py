"""Internal SSE streaming helpers — shared by client.py and repl.py."""

import json


def parse_sse_lines(resp):
    """Read SSE lines from an HTTP response, yield (token_text, is_last).

    Yields parsed (text, is_last) tuples from ``data: {...}`` events.
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
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                is_last = choices[0].get("finish_reason") is not None
                yield text, is_last
        except json.JSONDecodeError:
            pass
