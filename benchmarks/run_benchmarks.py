#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "You are benchmarking a local LLM on Raspberry Pi. "
    "Explain KV cache optimization, speculative decoding, and CPU profiling."
)


def run_command(cmd, timeout):
    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output": proc.stdout,
    }


def parse_llama_timings(output):
    parsed = {}
    patterns = {
        "prompt_tokens_per_sec": r"prompt eval time.*?/\s*([0-9.]+)\s*tokens per second",
        "generation_tokens_per_sec": r"eval time.*?/\s*([0-9.]+)\s*tokens per second",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            parsed[key] = float(matches[-1])
    return parsed


def append_result(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def benchmark_cli(args, label, extra_args, timeout=900):
    cmd = [
        str(args.llama_bin / "llama-cli"),
        "-m",
        str(args.target_model),
        "-p",
        DEFAULT_PROMPT,
        "-n",
        str(args.n_gen),
        "-t",
        str(args.threads),
        "--no-display-prompt",
        "--no-conversation",
        "--single-turn",
        "--show-timings",
    ] + extra_args

    result = run_command(cmd, timeout=timeout)
    record = {
        "label": label,
        "suite": "cli",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "metrics": parse_llama_timings(result["output"]),
        "returncode": result["returncode"],
        "elapsed_sec": result["elapsed_sec"],
        "command": " ".join(cmd),
    }
    append_result(args.output, record)
    print(json.dumps(record, indent=2))
    return result["returncode"] == 0


def benchmark_bench(args, label, extra_args, timeout=900):
    cmd = [
        str(args.llama_bin / "llama-bench"),
        "-m",
        str(args.target_model),
        "-p",
        str(args.n_prompt),
        "-n",
        str(args.n_gen),
        "-c",
        str(args.ctx),
        "-t",
        str(args.threads),
        "-o",
        "json",
    ] + extra_args

    result = run_command(cmd, timeout=timeout)
    record = {
        "label": label,
        "suite": "llama-bench",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "returncode": result["returncode"],
        "elapsed_sec": result["elapsed_sec"],
        "command": " ".join(cmd),
        "raw_output": result["output"].strip(),
    }
    append_result(args.output, record)
    print(json.dumps(record, indent=2))
    return result["returncode"] == 0


def existing_file(path, description):
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-bin", type=Path, default=ROOT / "vendor/llama.cpp/build/bin")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--target-model", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark_results.jsonl")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--n-prompt", type=int, default=256)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--suite", choices=["all", "bench", "kv", "threads", "speculative"], default="all")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.target_model = args.target_model or args.model_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    args.draft_model = args.draft_model or args.model_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

    existing_file(args.llama_bin / "llama-cli", "llama-cli")
    existing_file(args.llama_bin / "llama-bench", "llama-bench")
    existing_file(args.target_model, "target model")

    if args.quick:
        args.n_prompt = min(args.n_prompt, 128)
        args.n_gen = min(args.n_gen, 64)

    if args.suite in ("all", "bench"):
        benchmark_bench(args, "bench/f16", [])
        benchmark_bench(args, "bench/q8_kv", ["-ctk", "q8_0", "-ctv", "q8_0"])

    if args.suite in ("all", "kv"):
        kv_types = ["f16", "q8_0"] if args.quick else ["f16", "q8_0", "q4_0"]
        for kv_type in kv_types:
            benchmark_cli(args, f"kv/{kv_type}", ["--cache-type-k", kv_type, "--cache-type-v", kv_type])

    if args.suite in ("all", "threads"):
        original_threads = args.threads
        thread_counts = [1, 2, 4] if not args.quick else [1, 4]
        for threads in thread_counts:
            args.threads = threads
            benchmark_cli(args, f"threads/{threads}", ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        args.threads = original_threads

    if args.suite in ("all", "speculative"):
        existing_file(args.draft_model, "draft model")
        benchmark_cli(
            args,
            "speculative/draft-simple",
            [
                "--spec-type",
                "draft-simple",
                "--model-draft",
                str(args.draft_model),
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--cache-type-k-draft",
                "q8_0",
                "--cache-type-v-draft",
                "q8_0",
            ],
            timeout=1200,
        )


if __name__ == "__main__":
    main()
