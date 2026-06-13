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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._stream_utils import parse_sse_lines


def query_completions(prompt: str, max_tokens: int = 50, temperature: float = 0.0,
                      top_k: int = 0, seed: int | None = None,
                      host: str = "localhost", port: int = 8000) -> dict:
    """Send a /v1/completions request (non-streaming)."""
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


def query_completions_stream(prompt: str, max_tokens: int = 50, temperature: float = 0.0,
                             top_k: int = 0, seed: int | None = None,
                             host: str = "localhost", port: int = 8000):
    """Send a streaming /v1/completions request, yield tokens and usage.

    Yields (token_text, is_last, usage_dict) triples.
    usage_dict is None for regular token chunks, and a dict with
    prompt_tokens/completion_tokens/total_tokens/time_seconds for
    the final usage chunk.
    """
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "stream": True,
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
            yield from parse_sse_lines(resp)
    except urllib.error.HTTPError as e:
        yield f"[HTTP {e.code}] {e.read().decode()}", True, None
    except urllib.error.URLError as e:
        yield f"[Connection failed] {e.reason}", True, None


def _run_stream_completion(args):
    """Run a streaming completion, printing tokens as they arrive and usage at the end."""
    print(f"Prompt: {args.prompt}")
    print("Streaming: ", end="", flush=True)
    usage = None
    for token_text, is_last, chunk_usage in query_completions_stream(
        args.prompt, args.max_tokens, args.temperature,
        args.top_k, args.seed, args.host, args.port,
    ):
        if chunk_usage is not None:
            usage = chunk_usage
            break  # usage chunk always arrives after finish_reason, nothing more to read
        elif token_text:
            print(token_text, end="", flush=True)
    print()
    if usage:
        ttft = usage.get("ttft_ms")
        tpot = usage.get("tpot_ms")
        tps = usage.get("tokens_per_second")
        parts = [
            f"└─ {usage.get('prompt_tokens', '?')} prompt + "
            f"{usage.get('completion_tokens', '?')} = "
            f"{usage.get('total_tokens', '?')} tokens "
            f"({usage.get('time_seconds', '?')}s)",
        ]
        if ttft is not None:
            parts.append(f"TTFT {ttft}ms")
        if tpot is not None:
            parts.append(f"TPOT {tpot}ms")
        if tps is not None:
            parts.append(f"{tps} tok/s")
        print(" | ".join(parts))
    return 0


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


def query_chat_stream(messages: list[dict], max_tokens: int = 50, temperature: float = 0.0,
                      top_k: int = 0, seed: int | None = None,
                      host: str = "localhost", port: int = 8000):
    """Send a streaming /v1/chat/completions request, yield delta text."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "stream": True,
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
            for line in resp:
                line = line.decode().strip()
                if not line or line == "data: [DONE]":
                    if line == "data: [DONE]":
                        yield None, True, None
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    usage = data.get("usage")
                    if usage:
                        yield None, True, usage
                        return
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content, False, None
    except urllib.error.HTTPError as e:
        yield f"[HTTP {e.code}]", True, None
    except urllib.error.URLError as e:
        yield f"[Connection failed]", True, None


def _strip_stream_role_prefix(token_text: str, prefix_buffer: str, prefix_done: bool):
    """Strip a leading role prefix from streaming chat chunks.

    The server may split role prefixes across multiple SSE chunks, e.g.:
    ``ass`` → ``istant`` → ``Hello``. Buffer the initial chunks until we can
    determine whether they form one of the known role prefixes.

    Returns ``(clean_text, new_prefix_buffer, new_prefix_done)``.
    ``clean_text`` may be empty while still buffering.
    """
    prefixes = (
        "assistant\n", "assistant",
        "user\n", "user",
        "system\n", "system",
    )
    if prefix_done or not token_text:
        return token_text, prefix_buffer, prefix_done

    prefix_buffer += token_text
    # Prefer the longest prefix match. If the current buffer is exactly a bare
    # prefix like "assistant" but could still extend to "assistant\n", keep
    # buffering so we don't leak a leading newline in the next chunk.
    if any(prefix.startswith(prefix_buffer) and prefix != prefix_buffer for prefix in prefixes):
        return "", prefix_buffer, False

    matched_full = None
    for prefix in prefixes:
        if prefix_buffer.startswith(prefix):
            matched_full = prefix
            break

    if matched_full is not None:
        return prefix_buffer[len(matched_full):], "", True
    if any(prefix.startswith(prefix_buffer) for prefix in prefixes):
        # Still waiting to see whether this becomes a full role prefix
        return "", prefix_buffer, False
    # Not a role prefix; flush buffered content as-is
    return prefix_buffer, "", True


def _run_stream_chat(args):
    """Run a streaming chat, printing tokens as they arrive."""
    messages = [{"role": "user", "content": args.prompt}]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})
    print(f"User: {args.prompt}")
    print("AI: ", end="", flush=True)
    usage = None
    full_text = ""
    prefix_buffer = ""
    prefix_done = False
    for token_text, is_last, chunk_usage in query_chat_stream(
        messages, args.max_tokens, args.temperature,
        args.top_k, args.seed, args.host, args.port,
    ):
        if chunk_usage is not None:
            usage = chunk_usage
        elif token_text:
            clean, prefix_buffer, prefix_done = _strip_stream_role_prefix(
                token_text, prefix_buffer, prefix_done
            )
            if clean:
                print(clean, end="", flush=True)
                full_text += clean
    print()
    if usage:
        print(f"└─ {usage.get('prompt_tokens', '?')} prompt + "
              f"{usage.get('completion_tokens', '?')} = "
              f"{usage.get('total_tokens', '?')} tokens "
              f"({usage.get('time_seconds', '?')}s)")
    return 0


def _run_repl(args):
    """Interactive completions REPL — fresh completion per turn, no chat history."""
    mode = "streaming" if not args.no_stream else "batch"
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

        if not args.no_stream:
            usage = None
            for token_text, is_last, chunk_usage in query_completions_stream(
                prompt, args.max_tokens, args.temperature,
                args.top_k, args.seed, args.host, args.port,
            ):
                if chunk_usage is not None:
                    usage = chunk_usage
                    break
                elif token_text:
                    print(token_text, end="", flush=True)
            print()
            if usage:
                ttft = usage.get("ttft_ms")
                tpot = usage.get("tpot_ms")
                tps = usage.get("tokens_per_second")
                parts = [
                    f"└─ {usage.get('prompt_tokens', '?')} prompt + "
                    f"{usage.get('completion_tokens', '?')} = "
                    f"{usage.get('total_tokens', '?')} tokens "
                    f"({usage.get('time_seconds', '?')}s)",
                ]
                if ttft is not None:
                    parts.append(f"TTFT {ttft}ms")
                if tpot is not None:
                    parts.append(f"TPOT {tpot}ms")
                if tps is not None:
                    parts.append(f"{tps} tok/s")
                print(" | ".join(parts))
        else:
            result = query_completions(prompt, args.max_tokens, args.temperature,
                                       args.top_k, args.seed, args.host, args.port)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(result.get("text", ""))
        print()
    return 0


def main():
    p = argparse.ArgumentParser(description="SmolLM2 chat client")
    p.add_argument("--prompt",
                   help="One-shot prompt (for --completion or --chat)")
    p.add_argument("--chat", action="store_true",
                   help="One-shot chat via /v1/chat/completions")
    p.add_argument("--completion", action="store_true",
                   help="One-shot completion via /v1/completions")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive chat session (default)")
    p.add_argument("--repl", action="store_true",
                   help="Interactive completions REPL")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming (batch mode)")
    p.add_argument("--system", default=None,
                   help="System prompt for chat mode")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--raw", action="store_true", help="Print raw JSON response")
    args = p.parse_args()

    # Default to interactive chat when no mode flags are given
    no_mode = not (args.interactive or args.repl or args.chat or args.completion)
    if args.interactive or no_mode:
        sys.exit(_run_interactive(args))

    if args.repl:
        sys.exit(_run_repl(args))

    if not args.no_stream and args.chat:
        sys.exit(_run_stream_chat(args))

    if not args.no_stream and args.completion:
        sys.exit(_run_stream_completion(args))

    if args.completion:
        result = query_completions(args.prompt or "", args.max_tokens, args.temperature,
                                   args.top_k, args.seed, args.host, args.port)
        _print_completion_result(result, args.prompt or "", args.raw)

    elif args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt or "Hello"})
        result = query_chat(messages, args.max_tokens, args.temperature,
                            args.top_k, args.seed, args.host, args.port)
        _print_chat_result(result, args.raw)


def _run_interactive(args):
    """Run an interactive chat session. Returns 0 on success, 1 on error."""
    mode = "streaming" if not args.no_stream else "batch"
    print(f"\nInteractive chat ({mode}, Ctrl+D, Ctrl+C to exit)\n")
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
        print(f"[System] {args.system}\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        if not args.no_stream:
            print("AI:  ", end="", flush=True)
            usage = None
            full_text = ""
            prefix_buffer = ""
            prefix_done = False
            for token_text, is_last, chunk_usage in query_chat_stream(
                messages, args.max_tokens, args.temperature,
                args.top_k, args.seed, args.host, args.port,
            ):
                if chunk_usage is not None:
                    usage = chunk_usage
                elif token_text:
                    clean, prefix_buffer, prefix_done = _strip_stream_role_prefix(
                        token_text, prefix_buffer, prefix_done
                    )
                    if clean:
                        print(clean, end="", flush=True)
                        full_text += clean
            print()
            if usage:
                ttft = usage.get("ttft_ms", "?")
                tpot = usage.get("tpot_ms", "?")
                tps = usage.get("tokens_per_second", "?")
                print(f"     TTFT {ttft}ms  TPOT {tpot}ms  {tps} tok/s\n")
            messages.append({"role": "assistant", "content": full_text})
        else:
            result = query_chat(messages, args.max_tokens, args.temperature,
                                args.top_k, args.seed, args.host, args.port)
            if "error" in result:
                print(f"Error: {result['error']}")
                return 1
            assistant_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"AI:  {assistant_text}")
            usage = result.get("usage", {})
            if usage:
                tps = usage.get("tokens_per_second", 0)
                print(f"     {usage.get('completion_tokens', '?')} tok / {usage.get('time_seconds', '?')}s"
                      f"  ({tps} tok/s)\n")
            else:
                print()
            messages.append({"role": "assistant", "content": assistant_text})
    return 0


def _print_completion_result(result: dict, prompt: str, raw: bool):
    if raw:
        print(json.dumps(result, indent=2))
        return
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    usage = result.get("usage", {})
    print(f"Prompt:     {prompt}")
    print(f"Generated:  {result.get('text', '')}")
    print(f"Usage:      {usage.get('prompt_tokens', '?')} prompt + "
          f"{usage.get('completion_tokens', '?')} completion = "
          f"{usage.get('total_tokens', '?')} tokens "
          f"({usage.get('time_seconds', '?')}s)")


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
