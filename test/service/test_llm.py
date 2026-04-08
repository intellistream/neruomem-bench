#!/usr/bin/env python3
"""
test_llm.py — CLI smoke-test for the meta-llama/Llama-3.1-8B-Instruct vLLM endpoint.

Usage:
    python test/service/test_llm.py [OPTIONS]

Options:
    --host HOST        Server host  (default: 127.0.0.1, overrides .env)
    --port PORT        Server port  (default: DEPLOY_LLM_PORT from .env, else 18000)
    --prompt TEXT      User message to send  (default: built-in sample)
    --max-tokens N     Max tokens to generate  (default: 128)
    --temperature N    Sampling temperature    (default: 0.0)
    --timeout N        Request timeout in seconds  (default: 60)
    --stream           Use streaming mode (SSE)
    --verbose          Print full response JSON (non-stream mode only)

Exit codes:
    0  success
    1  connection / HTTP / validation error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ── Load .env from project root ───────────────────────────────────────────────
def _load_dotenv() -> dict[str, str]:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    result: dict[str, str] = {}
    if not env_file.exists():
        return result
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.split("#")[0].strip()
    return result


_ENV = _load_dotenv()

DEFAULT_PORT = int(_ENV.get("DEPLOY_LLM_PORT", "18000"))
DEFAULT_PROMPT = "What is retrieval-augmented generation (RAG)? Reply in two sentences."
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ── Health-check: wait for server to become ready ────────────────────────────
def wait_for_server(host: str, port: int, wait_timeout: int) -> bool:
    """Poll /health until 200 OK or wait_timeout seconds elapse."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + wait_timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[OK]    Server is ready (attempt {attempt}).")
                    return True
        except Exception:  # noqa: BLE001
            pass
        remaining = int(deadline - time.monotonic())
        print(f"[WAIT]  Server not ready yet — retrying... ({remaining}s left)",
              end="\r", flush=True)
        time.sleep(3)
    print()
    return False


# ── Non-streaming test ────────────────────────────────────────────────────────
def test_llm(
    host: str,
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    verbose: bool,
) -> bool:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    print(f"[INFO]  Target  : {url}")
    print(f"[INFO]  Prompt  : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"[FAIL]  Connection error: {exc.reason}")
        print(f"        Is the LLM server running on {host}:{port}?")
        print(f"        Start it with:  bash scripts/deploy/deploy_llama31_8b.sh")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL]  Unexpected error: {exc}")
        return False

    # Validate response shape
    choices = body.get("choices", [])
    if not choices:
        print("[FAIL]  Response contains no 'choices' field.")
        if verbose:
            print(json.dumps(body, indent=2))
        return False

    reply = choices[0].get("message", {}).get("content", "").strip()
    usage = body.get("usage", {})

    print(f"[OK]    Reply:")
    print()
    for line in reply.splitlines():
        print(f"        {line}")
    print()
    print(f"[OK]    Latency      : {elapsed * 1000:.1f} ms")
    print(f"[OK]    Tokens       : prompt={usage.get('prompt_tokens', '?')}, "
          f"completion={usage.get('completion_tokens', '?')}, "
          f"total={usage.get('total_tokens', '?')}")
    print(f"[OK]    Finish reason: {choices[0].get('finish_reason', '?')}")

    if verbose:
        print()
        print("[VERBOSE] Full response:")
        print(json.dumps(body, indent=2))

    return True


# ── Streaming test ────────────────────────────────────────────────────────────
def test_llm_stream(
    host: str,
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> bool:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    print(f"[INFO]  Target  : {url}  (streaming)")
    print(f"[INFO]  Prompt  : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()
    print("[OK]    Reply (streaming):")
    print()
    print("        ", end="", flush=True)

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    token_count = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    token_count += 1
    except urllib.error.URLError as exc:
        print()
        print(f"\n[FAIL]  Connection error: {exc.reason}")
        print(f"        Is the LLM server running on {host}:{port}?")
        print(f"        Start it with:  bash scripts/deploy/deploy_llama31_8b.sh")
        return False
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"\n[FAIL]  Unexpected error: {exc}")
        return False

    elapsed = time.perf_counter() - t0
    print()
    print()
    print(f"[OK]    Latency : {elapsed * 1000:.1f} ms  (~{token_count} chunks received)")
    return True


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Llama-3.1-8B-Instruct vLLM endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--wait", action="store_true",
                        help="Wait for server to become ready before testing")
    parser.add_argument("--wait-timeout", type=int, default=300, metavar="SECS",
                        help="Max seconds to wait for server (default: 300)")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print()
    print("  === neuromem-bench · LLM service test (Llama-3.1-8B-Instruct) ===")
    print()

    if args.wait:
        print(f"[INFO]  Waiting for server at {args.host}:{args.port} "
              f"(up to {args.wait_timeout}s)...")
        if not wait_for_server(args.host, args.port, args.wait_timeout):
            print(f"[FAIL]  Server did not become ready within {args.wait_timeout}s.")
            print("        Start it with:  bash scripts/deploy/deploy_llama31_8b.sh")
            sys.exit(1)
        print()

    if args.stream:
        ok = test_llm_stream(
            args.host, args.port, args.prompt,
            args.max_tokens, args.temperature, args.timeout,
        )
    else:
        ok = test_llm(
            args.host, args.port, args.prompt,
            args.max_tokens, args.temperature, args.timeout, args.verbose,
        )

    print()
    if ok:
        print("[PASS]  LLM service is healthy.")
    else:
        print("[FAIL]  LLM service test failed.")
    print()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
