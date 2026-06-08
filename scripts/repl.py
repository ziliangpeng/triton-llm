#!/usr/bin/env python3
"""Interactive REPL for SmolLM2 HTTP server — each turn is a fresh completion.

Usage:
    python scripts/repl.py
    python scripts/repl.py --port 8000 --max-tokens 50 --temperature 0.7
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


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


def main():
    p = argparse.ArgumentParser(description="REPL — fresh completion each turn")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    print(f"SmolLM2 REPL  (host={args.host}:{args.port}, max_tokens={args.max_tokens})")
    print("Type your prompt.  Ctrl+D / Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue

        result = query(prompt, args.max_tokens, args.temperature,
                       args.host, args.port)
        print(result)
        print()


if __name__ == "__main__":
    main()
