#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def command_arg(command, flag):
    parts = command.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
    return None


def kv_cache(command):
    k_type = command_arg(command, "--cache-type-k") or command_arg(command, "-ctk") or "f16"
    v_type = command_arg(command, "--cache-type-v") or command_arg(command, "-ctv") or "f16"
    return f"k={k_type}, v={v_type}"


def generation_tokens(record):
    value = command_arg(record.get("command", ""), "-n")
    return int(value) if value else None


def threads(record):
    value = command_arg(record.get("command", ""), "-t")
    return int(value) if value else None


def estimated_tps(record):
    metrics = record.get("metrics") or {}
    if "generation_tokens_per_sec" in metrics:
        return float(metrics["generation_tokens_per_sec"])
    tokens = generation_tokens(record)
    elapsed = float(record.get("elapsed_sec") or 0)
    if tokens and elapsed > 0:
        return tokens / elapsed
    return None


def latest_successful_cli(records):
    latest = {}
    for record in records:
        if record.get("suite") != "cli" or record.get("returncode") != 0:
            continue
        latest[record["label"]] = record
    return latest


def row_for(label, record):
    tps = estimated_tps(record)
    return {
        "label": label,
        "elapsed_sec": round(float(record.get("elapsed_sec") or 0), 3),
        "tokens": generation_tokens(record) or "",
        "estimated_tokens_per_sec": round(tps, 3) if tps is not None else "",
        "threads": threads(record) or "",
        "kv_cache": kv_cache(record.get("command", "")),
        "timestamp": record.get("timestamp", ""),
        "host": record.get("host", ""),
    }


def parse_perf_stat(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    patterns = {
        "cpus_utilized": r"#\s+([0-9.]+)\s+CPUs utilized",
        "ipc": r"#\s+([0-9.]+)\s+insn per cycle",
        "cpu_ghz": r"#\s+([0-9.]+)\s+GHz",
        "l1_miss_percent": r"#\s+([0-9.]+)% of all L1-dcache accesses",
        "elapsed_sec": r"([0-9.]+)\s+seconds time elapsed",
        "user_sec": r"([0-9.]+)\s+seconds user",
        "sys_sec": r"([0-9.]+)\s+seconds sys",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def markdown_table(rows):
    headers = ["label", "elapsed_sec", "estimated_tokens_per_sec", "threads", "kv_cache"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join(lines)


def speedup(base, candidate):
    if not base or not candidate:
        return None
    base_elapsed = float(base.get("elapsed_sec") or 0)
    candidate_elapsed = float(candidate.get("elapsed_sec") or 0)
    if base_elapsed <= 0 or candidate_elapsed <= 0:
        return None
    return base_elapsed / candidate_elapsed


def build_report(rows, records, perf):
    lines = ["# Pi Oven Benchmark Analytics", ""]
    lines.append("## Summary Table")
    lines.append("")
    lines.append(markdown_table(rows))
    lines.append("")

    if rows:
        fastest = max(rows, key=lambda row: float(row["estimated_tokens_per_sec"] or 0))
        slowest = min(rows, key=lambda row: float(row["estimated_tokens_per_sec"] or 0))
        lines.append("## Highlights")
        lines.append("")
        lines.append(
            f"- Fastest successful run: `{fastest['label']}` at "
            f"{fastest['estimated_tokens_per_sec']} estimated tokens/sec."
        )
        lines.append(
            f"- Slowest successful run: `{slowest['label']}` at "
            f"{slowest['estimated_tokens_per_sec']} estimated tokens/sec."
        )

    kv_f16 = records.get("kv/f16")
    kv_q8 = records.get("kv/q8_0")
    if kv_f16 and kv_q8:
        factor = speedup(kv_f16, kv_q8)
        if factor:
            pct = (factor - 1.0) * 100
            lines.append(
                f"- KV q8_0 vs f16 elapsed speedup: `{factor:.3f}x` "
                f"({pct:+.1f}%)."
            )

    thread_1 = records.get("threads/1")
    thread_4 = records.get("threads/4")
    if thread_1 and thread_4:
        factor = speedup(thread_1, thread_4)
        if factor:
            efficiency = factor / 4
            lines.append(
                f"- 4 threads vs 1 thread speedup: `{factor:.3f}x`; "
                f"parallel efficiency vs ideal 4x: `{efficiency:.3f}`."
            )

    spec = records.get("speculative/draft-simple")
    baseline = records.get("kv/q8_0") or records.get("threads/4")
    if spec and baseline:
        factor = speedup(baseline, spec)
        if factor:
            lines.append(
                f"- Speculative decoding vs q8 baseline speed factor: `{factor:.3f}x`; "
                "below 1.0 means speculative decoding was slower."
            )

    if perf:
        lines.append("")
        lines.append("## perf stat")
        lines.append("")
        for key, value in perf.items():
            lines.append(f"- `{key}`: `{value}`")

        ipc = perf.get("ipc")
        l1_miss = perf.get("l1_miss_percent")
        cpus = perf.get("cpus_utilized")
        if ipc is not None or l1_miss is not None or cpus is not None:
            lines.append("")
            lines.append("## Interpretation")
            lines.append("")
            if cpus is not None:
                lines.append(f"- CPU utilization was about `{cpus:.2f}` cores, so the 4-thread run is using parallel CPU.")
            if ipc is not None:
                lines.append(f"- IPC was `{ipc:.2f}`. Values below 1 often suggest stalls from memory, cache, or vectorization limits.")
            if l1_miss is not None:
                lines.append(f"- L1 data miss rate was `{l1_miss:.2f}%`, which is low; the bottleneck may be broader memory bandwidth or compute rather than L1 misses.")

    lines.append("")
    return "\n".join(lines)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/benchmark_results.jsonl")
    parser.add_argument("--perf", type=Path, default=ROOT / "results/perf_stat.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analytics")
    return parser.parse_args()


def main():
    args = parse_args()
    records = latest_successful_cli(load_jsonl(args.results))
    if not records:
        raise SystemExit(f"No successful llama-cli rows found in {args.results}")

    rows = [row_for(label, record) for label, record in records.items()]
    rows.sort(key=lambda row: row["label"])
    perf = parse_perf_stat(args.perf)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    report_path = args.output_dir / "report.md"
    write_csv(rows, csv_path)
    report_path.write_text(build_report(rows, records, perf), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print()
    print(build_report(rows, records, perf))


if __name__ == "__main__":
    main()
