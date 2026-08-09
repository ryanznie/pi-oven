#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import selectors
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "You are benchmarking a local LLM on Raspberry Pi. "
    "Explain KV cache optimization, speculative decoding, and CPU profiling."
)


def run_command(cmd, timeout, measure_ttft=False):
    if measure_ttft:
        return run_streaming_command(cmd, timeout)

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "returncode": 124,
            "elapsed_sec": round(time.time() - started, 3),
            "output": exc.stdout or f"Timed out after {timeout} seconds",
            "timed_out": True,
        }
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output": proc.stdout,
        "timed_out": False,
    }


def run_streaming_command(cmd, timeout):
    started = time.time()
    output_chunks = []
    ttft_sec = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")

    timed_out = False
    while selector.get_map():
        if time.time() - started > timeout:
            timed_out = True
            proc.kill()
            break

        events = selector.select(timeout=0.2)
        if not events and proc.poll() is not None:
            break

        for key, _ in events:
            chunk = key.fileobj.read1(4096) if hasattr(key.fileobj, "read1") else key.fileobj.read(4096)
            if not chunk:
                selector.unregister(key.fileobj)
                continue

            stream_name = key.data
            text = chunk.decode("utf-8", errors="replace")
            output_chunks.append(text)
            if stream_name == "stdout" and ttft_sec is None and text.strip():
                ttft_sec = round(time.time() - started, 3)

    try:
        returncode = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        timed_out = True

    output = "".join(output_chunks)
    if timed_out and not output:
        output = f"Timed out after {timeout} seconds"

    return {
        "command": cmd,
        "returncode": 124 if timed_out else returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output": output,
        "timed_out": timed_out,
        "ttft_sec": ttft_sec,
    }


def parse_llama_timings(output):
    parsed = {}
    rate_pattern = re.compile(r"([0-9.]+)\s*(?:tokens per second|tok/s)", re.IGNORECASE)
    time_pattern = re.compile(r"([0-9.]+)\s*ms", re.IGNORECASE)
    token_pattern = re.compile(r"/\s*([0-9]+)\s*(?:tokens|runs)", re.IGNORECASE)
    for line in output.splitlines():
        lower = line.lower()
        time_match = time_pattern.search(line)
        token_match = token_pattern.search(line)
        rate_match = rate_pattern.search(line)
        if "load time" in lower and time_match:
            parsed["load_time_ms"] = float(time_match.group(1))
        if "prompt eval time" in lower:
            if time_match:
                parsed["prompt_eval_ms"] = float(time_match.group(1))
            if token_match:
                parsed["prompt_tokens"] = int(token_match.group(1))
            if rate_match:
                parsed["prompt_tokens_per_sec"] = float(rate_match.group(1))
        elif "eval time" in lower:
            if time_match:
                parsed["generation_eval_ms"] = float(time_match.group(1))
            if token_match:
                parsed["generation_tokens_or_runs"] = int(token_match.group(1))
            if rate_match:
                parsed["generation_tokens_per_sec"] = float(rate_match.group(1))
    return parsed


def timing_lines(output):
    return [
        line
        for line in output.splitlines()
        if "eval time" in line.lower() or "tokens per second" in line.lower()
    ]


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

    print(f"\n==> Running {label} ({args.n_gen} generated tokens, {args.threads} threads)", flush=True)
    result = run_command(cmd, timeout=timeout, measure_ttft=True)
    record = {
        "label": label,
        "suite": "cli",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "metrics": parse_llama_timings(result["output"]),
        "timing_lines": timing_lines(result["output"]),
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "ttft_sec": result.get("ttft_sec"),
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
        "-r",
        "1",
        "--no-warmup",
        "--progress",
        "-p",
        str(args.n_prompt),
        "-n",
        str(args.n_gen),
        "-t",
        str(args.threads),
        "-o",
        "json",
    ] + extra_args

    print(f"\n==> Running {label} with llama-bench", flush=True)
    result = run_command(cmd, timeout=timeout)
    record = {
        "label": label,
        "suite": "llama-bench",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
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
    parser.add_argument("--skip-bench", action="store_true", help="Skip llama-bench runs.")
    parser.add_argument("--include-bench", action="store_true", help="Include llama-bench even in --quick mode.")
    parser.add_argument("--skip-speculative", action="store_true", help="Skip the slow draft-model run.")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds before a single run is marked timed out.")
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
        if args.suite == "all" and not args.include_bench:
            args.skip_bench = True
            print("Quick mode skips llama-bench by default; use --include-bench or --suite bench to run it.", flush=True)

    if args.suite in ("all", "bench") and not args.skip_bench:
        benchmark_bench(args, "bench/f16", [], timeout=args.timeout)
        benchmark_bench(args, "bench/q8_kv", ["-ctk", "q8_0", "-ctv", "q8_0"], timeout=args.timeout)

    if args.suite in ("all", "kv"):
        kv_types = ["f16", "q8_0"] if args.quick else ["f16", "q8_0", "q4_0"]
        for kv_type in kv_types:
            benchmark_cli(
                args,
                f"kv/{kv_type}",
                ["--cache-type-k", kv_type, "--cache-type-v", kv_type],
                timeout=args.timeout,
            )

    if args.suite in ("all", "threads"):
        original_threads = args.threads
        thread_counts = [1, 2, 4] if not args.quick else [1, 4]
        for threads in thread_counts:
            args.threads = threads
            benchmark_cli(
                args,
                f"threads/{threads}",
                ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
                timeout=args.timeout,
            )
        args.threads = original_threads

    if args.suite in ("all", "speculative") and not args.skip_speculative:
        existing_file(args.draft_model, "draft model")
        print("Speculative decoding was slow in the first Pi run; use --skip-speculative for fast iteration.", flush=True)
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
            timeout=max(args.timeout, 1200),
        )


if __name__ == "__main__":
    main()
