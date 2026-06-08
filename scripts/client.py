#!/usr/bin/env python3
"""Client for the SmolLM2 HTTP inference server.

Usage:
    # One-shot completion:
    python scripts/client.py --prompt "The capital of France is" --max-tokens 20

    # Chat-style:
    python scripts/client.py --chat --prompt "Hello, what can you do?"

    # Interactive session:
    python scripts/client.py --interactive

SSH tunnel setup (run this in another terminal before using the client):

    1. Start the server on the remote GPU node:
       ssh <login-host> <your-srun-command> python scripts/serve_model.py --port 8000

    2. Forward the port to your local machine:
       ssh -L 8000:localhost:8000 -N <login-host>

    3. In a third terminal, use this client:
       python scripts/client.py --prompt "Hello world"

    If the server is behind an intermediate jump host, adjust the tunnel:
       ssh -J <jump-host> -L 8000:localhost:8000 -N <compute-host>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def query_completions(prompt: str, max_tokens: int = 50, temperature: float = 0.0,
                      top_k: int = 0, seed: int | None = None,
                      host: str = "localhost", port: int = 8000) -> dict:
    """Send a /v1/completions request."""
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


def query_chat(messages: list[dict], max_tokens: int = 50, temperature: float = 0.0,
               top_k: int = 0, seed: int | None = None,
               host: str = "localhost", port: int = 8000) -> dict:
    """Send a /v1/chat/completions request."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    if seed is not None:
        body["seed"] = seed

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
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
    p.add_argument("--prompt", default="The capital of France is",
                   help="Input prompt (only used with --chat for the user message)")
    p.add_argument("--chat", action="store_true",
                   help="Use /v1/chat/completions endpoint")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive chat session (implies --chat)")
    p.add_argument("--system", default=None,
                   help="System prompt for chat mode")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--raw", action="store_true", help="Print raw JSON response")
    args = p.parse_args()

    if args.interactive:
        _run_interactive(args)
        return

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        result = query_chat(messages, args.max_tokens, args.temperature,
                            args.top_k, args.seed, args.host, args.port)
        _print_chat_result(result, args.raw)
    else:
        result = query_completions(args.prompt, args.max_tokens, args.temperature,
                                   args.top_k, args.seed, args.host, args.port)
        _print_completion_result(result, args.raw)


def _run_interactive(args):
    """Run an interactive chat session."""
    print("Interactive chat (Ctrl+D to exit)\n")
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
        print(f"[System] {args.system}\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print()
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        result = query_chat(messages, args.max_tokens, args.temperature,
                            args.top_k, args.seed, args.host, args.port)

        if "error" in result:
            print(f"Error: {result['error']}")
            break

        assistant_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"AI:  {assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})


def _print_completion_result(result: dict, raw: bool):
    if raw:
        print(json.dumps(result, indent=2))
        return
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Prompt:     {sys.argv[sys.argv.index('--prompt') + 1] if '--prompt' in sys.argv else ''}")
    print(f"Generated:  {result['text']}")
    print(f"Usage:      {result['usage']['prompt_tokens']} prompt + "
          f"{result['usage']['completion_tokens']} completion = "
          f"{result['usage']['total_tokens']} tokens "
          f"({result['usage']['time_seconds']}s)")


def _print_chat_result(result: dict, raw: bool):
    if raw:
        print(json.dumps(result, indent=2))
        return
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    for choice in result.get("choices", []):
        msg = choice.get("message", {})
        print(f"{msg.get('role', 'assistant')}: {msg.get('content', '')}")
    usage = result.get("usage", {})
    print(f"\n(usage: {usage.get('prompt_tokens', '?')} prompt + "
          f"{usage.get('completion_tokens', '?')} completion = "
          f"{usage.get('total_tokens', '?')} tokens "
          f"({usage.get('time_seconds', '?')}s))")


if __name__ == "__main__":
    main()
