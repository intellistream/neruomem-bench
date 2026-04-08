#!/usr/bin/env python3
"""
neuromem-bench CLI entry point.

Usage:
    neuromem-bench <command> [args...]

Commands:
    test embed   [OPTIONS]   Smoke-test the BAAI/bge-m3 embedding service
    test llm     [OPTIONS]   Smoke-test the Llama-3.1-8B-Instruct LLM service

Run `neuromem-bench test embed --help` or `neuromem-bench test llm --help`
for per-command options.
"""
from __future__ import annotations

import sys


def _usage() -> None:
    print(__doc__)


def main() -> None:
    args = sys.argv[1:]

    if not args:
        _usage()
        sys.exit(0)

    # ── neuromem-bench test <subcommand> ──────────────────────────────────────
    if args[0] == "test":
        if len(args) < 2:
            print("Usage: neuromem-bench test {embed|llm} [OPTIONS]")
            sys.exit(1)

        subcommand = args[1].lower()
        # Rewrite sys.argv so the sub-module's argparse sees only its own args
        sys.argv = [f"neuromem-bench test {subcommand}"] + args[2:]

        if subcommand == "embed":
            from test.service.test_embed import main as _main
            _main()
        elif subcommand == "llm":
            from test.service.test_llm import main as _main
            _main()
        else:
            print(f"[ERROR] Unknown test subcommand: '{subcommand}'")
            print("        Available: embed, llm")
            sys.exit(1)

    else:
        print(f"[ERROR] Unknown command: '{args[0]}'")
        _usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
