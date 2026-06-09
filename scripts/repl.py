#!/usr/bin/env python3
"""Interactive REPL for SmolLM2 HTTP server — each turn is a fresh completion.

Usage:
    python scripts/repl.py
    python scripts/repl.py --port 8000 --max-tokens 50 --temperature 0.7
    python scripts/repl.py --stream
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._stream_utils import parse_sse_lines


def query(prompt: str, max_tokens: int, temperature: float,
          host: str, port: int) -> str:
    """Send /v1/completions, return generated text."""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return f"[HTTP {e.code}] {e.read().decode()}"
    except urllib.error.URLError as e:
        return f"[Connection failed] {e.reason}"

    return data.get("text", json.dumps(data))


def query_stream(prompt: str, max_tokens: int, temperature: float,
                 host: str, port: int):
    """Send streaming /v1/completions, yield token texts as they arrive."""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            yield from parse_sse_lines(resp)
    except urllib.error.HTTPError as e:
        yield f"[HTTP {e.code}] {e.read().decode()}", True
    except urllib.error.URLError as e:
        yield f"[Connection failed] {e.reason}", True


def main():
    p = argparse.ArgumentParser(description="REPL — fresh completion each turn")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens as they are generated")
    args = p.parse_args()

    mode = "streaming" if args.stream else "batch"
    print(f"SmolLM2 REPL  (host={args.host}:{args.port}, max_tokens={args.max_tokens}, mode={mode})")
    print("Type your prompt.  Ctrl+D / Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue

        if args.stream:
            for token_text, is_last in query_stream(
                prompt, args.max_tokens, args.temperature, args.host, args.port,
            ):
                if is_last and not token_text:
                    break
                print(token_text, end="", flush=True)
            print()
        else:
            result = query(prompt, args.max_tokens, args.temperature,
                           args.host, args.port)
            print(result)
        print()


if __name__ == "__main__":
    main()
