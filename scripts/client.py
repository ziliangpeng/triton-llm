#!/usr/bin/env python3
"""Client for SmolLM2 Triton HTTP server.

Usage:
    python scripts/client.py --prompt "The capital of France is" --max-tokens 20

Requires SSH tunnel to GPU compute node:
    ssh -J gcp5 -L 8000:localhost:8000 -N gcp5-h100-0-9 &
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def query(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
    host: str = "localhost",
    port: int = 8000,
) -> dict:
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    if seed is not None:
        body["seed"] = seed

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


def main():
    p = argparse.ArgumentParser(description="Query SmolLM2 HTTP server")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--raw", action="store_true", help="Print raw JSON response")
    args = p.parse_args()

    result = query(args.prompt, args.max_tokens, args.temperature,
                   args.top_k, args.seed, args.host, args.port)

    if args.raw:
        print(json.dumps(result, indent=2))
        return

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Prompt:     {args.prompt}")
    print(f"Generated:  {result['text']}")
    print(f"Usage:      {result['usage']['prompt_tokens']} prompt + "
          f"{result['usage']['completion_tokens']} completion = "
          f"{result['usage']['total_tokens']} tokens "
          f"({result['usage']['time_seconds']}s)")


if __name__ == "__main__":
    main()
