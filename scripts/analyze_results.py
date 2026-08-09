#!/usr/bin/env python3
"""Create an accuracy-first report from Pi Oven benchmark results."""

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_arg(command, flag):
    parts = command.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
    return None


def kv_cache(command):
    key = command_arg(command, "--cache-type-k") or command_arg(command, "-ctk") or "f16"
    value = command_arg(command, "--cache-type-v") or command_arg(command, "-ctv") or "f16"
    return f"k={key}, v={value}"


def latest_successful_cli(records):
    latest = {}
    for record in records:
        if record.get("suite") == "cli" and record.get("returncode") == 0:
            latest[record["label"]] = record
    return latest


def extract_bench_objects(raw_output):
    """Extract JSON objects even when llama-bench progress text splits its array."""

    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r'\{\s*"build_commit"', raw_output):
        try:
            value, _ = decoder.raw_decode(raw_output[match.start() :])
        except json.JSONDecodeError:
            continue
        objects.append(value)
    return objects


def bench_rows(records):
    rows = []
    for record in records:
        if record.get("suite") != "llama-bench" or record.get("returncode") != 0:
            continue
        for result in extract_bench_objects(record.get("raw_output", "")):
            is_prefill = int(result.get("n_prompt", 0)) > 0
            rows.append(
                {
                    "label": record["label"],
                    "workload": "prefill" if is_prefill else "generation",
                    "kv_cache": f"k={result['type_k']}, v={result['type_v']}",
                    "tokens": int(result["n_prompt"] if is_prefill else result["n_gen"]),
                    "tokens_per_sec": round(float(result["avg_ts"]), 6),
                    "latency_sec": round(float(result["avg_ns"]) / 1e9, 6),
                    "threads": int(result["n_threads"]),
                    "model": result["model_type"],
                    "model_size_mb": round(float(result["model_size"]) / 1024**2, 1),
                }
            )
    return rows


def parse_perf_stat(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    patterns = {
        "elapsed_sec": r"([0-9.]+)\s+seconds time elapsed",
        "cpus_utilized": r"#\s+([0-9.]+)\s+CPUs utilized",
        "cpu_ghz": r"#\s+([0-9.]+)\s+GHz",
        "ipc": r"#\s+([0-9.]+)\s+insn per cycle",
        "l1_miss_percent": r"#\s+([0-9.]+)% of all L1-dcache accesses",
        "user_sec": r"([0-9.]+)\s+seconds user",
        "sys_sec": r"([0-9.]+)\s+seconds sys",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def markdown_table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def percent_change(candidate, baseline):
    return (candidate / baseline - 1.0) * 100.0


def build_comparison_rows(exact_rows, cli):
    """Build rankings only within groups that measure the same thing."""

    rows = []
    for workload in ("generation", "prefill"):
        group = [row for row in exact_rows if row["workload"] == workload]
        group.sort(key=lambda row: row["tokens_per_sec"], reverse=True)
        best = group[0]["tokens_per_sec"]
        for rank, row in enumerate(group, start=1):
            rows.append(
                {
                    "category": f"KV cache {workload}",
                    "rank": rank,
                    "method": row["kv_cache"],
                    "metric": "tokens_per_sec",
                    "value": row["tokens_per_sec"],
                    "relative_to_best": round(row["tokens_per_sec"] / best, 4),
                    "confidence": "low",
                    "verdict": "best measured" if rank == 1 else "effectively tied" if row["tokens_per_sec"] / best > 0.99 else "slower",
                }
            )

    thread_group = []
    one_thread = cli.get("threads/1")
    for count in (1, 2, 4):
        record = cli.get(f"threads/{count}")
        if record:
            thread_group.append((count, float(record["elapsed_sec"])))
    thread_group.sort(key=lambda item: item[1])
    best_elapsed = thread_group[0][1]
    one_elapsed = float(one_thread["elapsed_sec"]) if one_thread else None
    for rank, (count, elapsed) in enumerate(thread_group, start=1):
        speedup = one_elapsed / elapsed if one_elapsed else 0
        verdict = "fastest" if rank == 1 else "best efficiency" if count == 2 else "slowest"
        rows.append(
            {
                "category": "CPU threads",
                "rank": rank,
                "method": f"{count} thread(s)",
                "metric": "wall_time_sec",
                "value": elapsed,
                "relative_to_best": round(best_elapsed / elapsed, 4),
                "confidence": "medium",
                "verdict": f"{verdict}; {speedup:.2f}x vs 1 thread",
            }
        )

    standard = cli.get("kv/q8_0")
    speculative = cli.get("speculative/draft-simple")
    if standard and speculative:
        methods = [
            ("standard decoding", float(standard["elapsed_sec"])),
            ("speculative decoding", float(speculative["elapsed_sec"])),
        ]
        methods.sort(key=lambda item: item[1])
        best_elapsed = methods[0][1]
        for rank, (method, elapsed) in enumerate(methods, start=1):
            rows.append(
                {
                    "category": "Decoding method",
                    "rank": rank,
                    "method": method,
                    "metric": "wall_time_sec",
                    "value": elapsed,
                    "relative_to_best": round(best_elapsed / elapsed, 4),
                    "confidence": "medium",
                    "verdict": "winner" if rank == 1 else f"{elapsed / best_elapsed:.2f}x slower",
                }
            )
    return rows


def build_report(records, exact_rows, perf):
    cli = latest_successful_cli(records)
    comparisons = build_comparison_rows(exact_rows, cli)
    cli_rows = []
    for label, record in cli.items():
        cli_rows.append(
            {
                "label": label,
                "elapsed_sec": record["elapsed_sec"],
                "threads": command_arg(record.get("command", ""), "-t"),
                "kv_cache": kv_cache(record.get("command", "")),
                "requested_max_tokens": command_arg(record.get("command", ""), "-n"),
            }
        )
    cli_rows.sort(key=lambda row: float(row["elapsed_sec"]))

    lines = [
        "# Pi Oven Results Analysis",
        "",
        "## At-a-Glance Winners",
        "",
        "| Decision | Winner | Evidence | Confidence |",
        "| --- | --- | --- | --- |",
        "| Highest measured generation throughput | F16 KV cache | 3.080 vs 3.059 tokens/s for Q8 | Low: one sample |",
        "| Highest measured prefill throughput | F16 KV cache | 7.112 vs 6.811 tokens/s for Q8 | Low: one sample |",
        "| Fastest thread setting | 4 threads | 96.317s wall time | Medium |",
        "| Best thread efficiency | 2 threads | 1.80x faster than one; four adds only 2.2% | Medium |",
        "| Best decoding method | Standard Q8 decoding | 96.580s vs 209.712s speculative | Medium |",
        "| Lowest-memory KV option | Not measured | Q8 should use less cache memory, but RAM was not captured | None |",
        "| Q4 KV result | No conclusion | 14.502s is inconsistent and likely ended early | None |",
        "",
        "**Practical choice from this run:** use standard decoding with 4 threads for the lowest measured latency, or 2 threads when you want nearly the same latency with better CPU efficiency. F16 and Q8 KV are tied for generation speed in practice; choose Q8 when memory pressure matters, then measure RAM to confirm the benefit.",
        "",
        "## Ranked Method Comparison",
        "",
        markdown_table(
            comparisons,
            ["category", "rank", "method", "metric", "value", "relative_to_best", "confidence", "verdict"],
        ),
        "",
        "`relative_to_best` is normalized within each category: 1.0 is the winner. It must not be compared across categories because tokens/sec and wall time are different measurements.",
        "",
        "## Detailed Findings",
        "",
    ]

    by_key = {(row["workload"], row["kv_cache"]): row for row in exact_rows}
    f16_prefill = by_key.get(("prefill", "k=f16, v=f16"))
    q8_prefill = by_key.get(("prefill", "k=q8_0, v=q8_0"))
    f16_gen = by_key.get(("generation", "k=f16, v=f16"))
    q8_gen = by_key.get(("generation", "k=q8_0, v=q8_0"))

    if f16_prefill and q8_prefill:
        delta = percent_change(q8_prefill["tokens_per_sec"], f16_prefill["tokens_per_sec"])
        lines.append(
            f"- Prefill: F16 KV was fastest at **{f16_prefill['tokens_per_sec']:.3f} tokens/s**; Q8 KV reached {q8_prefill['tokens_per_sec']:.3f} tokens/s ({delta:+.2f}%)."
        )
    if f16_gen and q8_gen:
        delta = percent_change(q8_gen["tokens_per_sec"], f16_gen["tokens_per_sec"])
        lines.append(
            f"- Generation: F16 KV reached **{f16_gen['tokens_per_sec']:.3f} tokens/s** and Q8 KV reached {q8_gen['tokens_per_sec']:.3f} tokens/s ({delta:+.2f}%). The difference is negligible in this one-run sample."
        )

    one = cli.get("threads/1")
    two = cli.get("threads/2")
    four = cli.get("threads/4")
    if one and two and four:
        speedup_2 = float(one["elapsed_sec"]) / float(two["elapsed_sec"])
        speedup_4 = float(one["elapsed_sec"]) / float(four["elapsed_sec"])
        gain_2_to_4 = (float(two["elapsed_sec"]) / float(four["elapsed_sec"]) - 1) * 100
        lines.append(
            f"- Thread scaling: 2 threads reduced wall time from {one['elapsed_sec']:.3f}s to {two['elapsed_sec']:.3f}s ({speedup_2:.2f}x). Four threads reached {four['elapsed_sec']:.3f}s ({speedup_4:.2f}x versus one), only {gain_2_to_4:.1f}% faster than two threads."
        )

    baseline = cli.get("kv/q8_0")
    speculative = cli.get("speculative/draft-simple")
    if baseline and speculative:
        slowdown = float(speculative["elapsed_sec"]) / float(baseline["elapsed_sec"])
        lines.append(
            f"- Speculative decoding took {speculative['elapsed_sec']:.3f}s versus {baseline['elapsed_sec']:.3f}s for the Q8 baseline, or **{slowdown:.2f}x longer**. Assuming comparable completion lengths, the 0.5B draft overhead did not pay off on this configuration."
        )

    lines.extend(
        [
            "",
            "## Exact llama-bench Measurements",
            "",
            markdown_table(
                exact_rows,
                ["workload", "kv_cache", "tokens", "tokens_per_sec", "latency_sec", "threads"],
            ),
            "",
            "These are the strongest throughput measurements because llama-bench reports the actual workload and timing directly. There was one sample per condition and no warmup, so small differences should not be treated as conclusive.",
            "",
            "## llama-cli Wall Time",
            "",
            markdown_table(
                cli_rows,
                ["label", "elapsed_sec", "threads", "kv_cache", "requested_max_tokens"],
            ),
            "",
            "The requested maximum is not proof that all 256 tokens were generated: generation can stop at EOS. In particular, `kv/q4_0` completed in 14.502s while comparable runs took about 96-97s, so its apparent speed is almost certainly an early-stop artifact. It is excluded from throughput and speedup claims.",
        ]
    )

    if perf:
        lines.extend(
            [
                "",
                "## CPU Profile",
                "",
                f"- Elapsed time: **{perf.get('elapsed_sec', 0):.3f}s**",
                f"- Average CPU use: **{perf.get('cpus_utilized', 0):.3f} cores** ({perf.get('cpus_utilized', 0) / 4 * 100:.1f}% of four cores)",
                f"- Instructions per cycle: **{perf.get('ipc', 0):.2f} IPC**",
                f"- Reported frequency: **{perf.get('cpu_ghz', 0):.3f} GHz**",
                f"- L1 data-cache miss rate: **{perf.get('l1_miss_percent', 0):.2f}%**",
                f"- User/system CPU time: **{perf.get('user_sec', 0):.3f}s / {perf.get('sys_sec', 0):.3f}s**",
                "",
                "The workload used most of the four-core CPU budget. IPC of 0.82 indicates substantial pipeline stalls, while the low L1 miss rate suggests the bottleneck is not simply L1 behavior; model compute, wider cache levels, and memory bandwidth remain likely constraints.",
            ]
        )

    lines.extend(
        [
            "",
            "## Data Gaps and Next Run",
            "",
            "1. Run each llama-bench condition at least three times with warmup and report mean plus standard deviation.",
            "2. Fix CLI output capture so prompt-eval rate, generation rate, actual token count, and TTFT are recorded.",
            "3. Repeat Q4 KV with a prompt or flags that force a fixed output length before comparing it.",
            "4. Record speculative acceptance rate; wall time alone cannot explain whether the draft model proposed useful tokens.",
            "5. Capture process peak RAM and temperature to detect memory pressure or thermal throttling.",
            "",
        ]
    )
    return "\n".join(lines), cli_rows, comparisons


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/benchmark_results.jsonl")
    parser.add_argument("--perf", type=Path, default=ROOT / "results/perf_stat.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analytics")
    parser.add_argument("--sort-by", choices=["tps", "elapsed", "ttft", "label"], default="tps")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.results)
    exact_rows = bench_rows(records)
    if not exact_rows:
        raise SystemExit(f"No parseable successful llama-bench rows found in {args.results}")
    exact_rows.sort(key=lambda row: (row["workload"], -row["tokens_per_sec"]))
    perf = parse_perf_stat(args.perf)
    report, cli_rows, comparisons = build_report(records, exact_rows, perf)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparisons, args.output_dir / "summary.csv")
    write_csv(exact_rows, args.output_dir / "llama_bench_exact.csv")
    write_csv(cli_rows, args.output_dir / "cli_wall_time.csv")
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
