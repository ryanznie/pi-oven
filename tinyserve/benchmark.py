#!/usr/bin/env python3
"""Run and record the TinyServe Phase 1 benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import torch

from tinyserve.models import DEFAULT_MODEL, load_model
from tinyserve.naive import generate_naive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "Explain why a KV cache makes autoregressive decoding faster."
CSV_FIELDS = [
    "timestamp",
    "host",
    "machine",
    "engine",
    "mode",
    "model",
    "weight_format",
    "dtype",
    "device",
    "threads",
    "prompt_tokens",
    "generated_tokens",
    "ttft_sec",
    "total_latency_sec",
    "tokens_per_sec",
    "forward_passes",
    "total_input_tokens_processed",
    "forward_input_tokens",
    "prompt",
    "generated_text",
]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TinyServe Phase 1: naive greedy autoregressive decoding."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet-steps", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "tinyserve_phase1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    print(f"Loading {args.model} as {args.dtype} on {args.device} ...", flush=True)
    tokenizer, model = load_model(args.model, args.dtype, args.device)

    print("\nNaive decoding (KV cache disabled):", flush=True)
    generated_text, stats = generate_naive(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        show_steps=not args.quiet_steps,
    )

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "engine": "tinyserve-pytorch",
        "mode": "naive",
        "model": args.model,
        "weight_format": "huggingface",
        "dtype": args.dtype,
        "device": args.device,
        "threads": args.threads,
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": stats.generated_tokens,
        "ttft_sec": f"{stats.ttft_sec:.6f}",
        "total_latency_sec": f"{stats.total_latency_sec:.6f}",
        "tokens_per_sec": f"{stats.tokens_per_sec:.6f}",
        "forward_passes": len(stats.forward_input_tokens),
        "total_input_tokens_processed": stats.total_input_tokens_processed,
        "forward_input_tokens": " ".join(map(str, stats.forward_input_tokens)),
        "prompt": args.prompt,
        "generated_text": generated_text.strip(),
    }
    append_csv(args.output, row)

    print("\nResult")
    print(f"  generated text: {generated_text.strip()!r}")
    print(f"  prompt tokens: {stats.prompt_tokens}")
    print(f"  generated tokens: {stats.generated_tokens}")
    print(f"  TTFT: {stats.ttft_sec:.3f} s")
    print(f"  total latency: {stats.total_latency_sec:.3f} s")
    print(f"  throughput: {stats.tokens_per_sec:.3f} tokens/s")
    print(f"  forward input lengths: {stats.forward_input_tokens}")
    print(f"  total input tokens processed: {stats.total_input_tokens_processed}")
    print(f"  CSV: {args.output}")


if __name__ == "__main__":
    main()
