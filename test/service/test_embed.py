#!/usr/bin/env python3
"""
test_embed.py — CLI smoke-test for the BAAI/bge-m3 vLLM embedding endpoint.

Usage:
    python test/service/test_embed.py [OPTIONS]

Options:
    --host HOST      Server host  (default: 127.0.0.1, overrides .env)
    --port PORT      Server port  (default: DEPLOY_EMBED_PORT from .env, else 18001)
    --texts TEXT...  One or more strings to embed  (default: built-in sample)
    --timeout N      Request timeout in seconds     (default: 30)
    --verbose        Print full response JSON

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

DEFAULT_PORT = int(_ENV.get("DEPLOY_EMBED_PORT", "18001"))
DEFAULT_TEXTS = [
    "neuromem is a memory management library for LLM applications.",
    "BAAI/bge-m3 supports dense, sparse and multi-vector retrieval.",
]


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


# ── Core test logic ───────────────────────────────────────────────────────────
def test_embed(host: str, port: int, texts: list[str], timeout: int, verbose: bool) -> bool:
    url = f"http://{host}:{port}/v1/embeddings"
    payload = json.dumps({"model": "BAAI/bge-m3", "input": texts}).encode()

    print(f"[INFO]  Target  : {url}")
    print(f"[INFO]  Texts   : {len(texts)} item(s)")
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
        print(f"        Is the embedding server running on {host}:{port}?")
        print(f"        Start it with:  bash scripts/deploy/deploy_bge_m3.sh")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL]  Unexpected error: {exc}")
        return False

    # Validate response shape
    data = body.get("data", [])
    if not data:
        print(f"[FAIL]  Response contains no 'data' field.")
        if verbose:
            print(json.dumps(body, indent=2))
        return False

    for i, item in enumerate(data):
        vec = item.get("embedding", [])
        print(f"[OK]    Text[{i}]  →  dim={len(vec)},  "
              f"head=[{', '.join(f'{v:.4f}' for v in vec[:5])} ...]")

    usage = body.get("usage", {})
    print()
    print(f"[OK]    Latency : {elapsed * 1000:.1f} ms")
    print(f"[OK]    Tokens  : prompt={usage.get('prompt_tokens', '?')}, "
          f"total={usage.get('total_tokens', '?')}")

    if verbose:
        print()
        print("[VERBOSE] Full response:")
        print(json.dumps(body, indent=2))

    return True


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the BAAI/bge-m3 vLLM embedding endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--texts", nargs="+", default=DEFAULT_TEXTS, metavar="TEXT")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--wait", action="store_true",
                        help="Wait for server to become ready before testing")
    parser.add_argument("--wait-timeout", type=int, default=300, metavar="SECS",
                        help="Max seconds to wait for server (default: 300)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print()
    print("  === neuromem-bench · Embedding service test (BAAI/bge-m3) ===")
    print()

    if args.wait:
        print(f"[INFO]  Waiting for server at {args.host}:{args.port} "
              f"(up to {args.wait_timeout}s)...")
        if not wait_for_server(args.host, args.port, args.wait_timeout):
            print(f"[FAIL]  Server did not become ready within {args.wait_timeout}s.")
            print("        Start it with:  bash scripts/deploy/deploy_bge_m3.sh")
            sys.exit(1)
        print()

    ok = test_embed(args.host, args.port, args.texts, args.timeout, args.verbose)

    print()
    if ok:
        print("[PASS]  Embedding service is healthy.")
    else:
        print("[FAIL]  Embedding service test failed.")
    print()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
