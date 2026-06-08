#!/usr/bin/env python3
"""Client for SmolLM2 Triton HTTP server.

Usage:
    # Set up tunnel (one-time):
    ssh -J gcp5 -L 8000:localhost:8000 -N gcp5-h100-0-9 &

    # Then query:
    python scripts/client.py --prompt "Hello world" --max-tokens 20
"""

import argparse
import json
import urllib.request
import urllib.error


def query(prompt: str, max_tokens: int = 50, temperature: float = 0.0,
          top_k: int = 0, seed: int | None = None,
          host: str = "localhost", port: int = 8000) -> dict:
    """Send a completion request to the server."""
    url = f"http://{host}:{port}/v1/completions"
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    if seed is not None:
        body["seed"] = seed

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


def main():
    p = argparse.ArgumentParser(description="Query SmolLM2 HTTP server")
    p.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    p.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--host", default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    args = p.parse_args()

    result = query(args.prompt, args.max_tokens, args.temperature,
                   args.top_k, args.seed, args.host, args.port)

    if "error" in result:
        print(f"❌ {result['error']}")
        return

    print(f"Prompt: {args.prompt}")
    if result.get("text"):
        print(f"Generated: {result['text']}")
    print(f"Usage: {result.get('usage', {})}")


if __name__ == "__main__":
    main()
