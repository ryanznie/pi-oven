#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import xy


ROOT = Path(__file__).resolve().parents[1]


THEME = xy.theme(
    background="#ffffff",
    plot_background="#fbfaf7",
    grid_color="#dedbd2",
    axis_color="#77736a",
    text_color="#171717",
    palette=["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"],
)


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


def estimate_tokens_per_second(record):
    metrics = record.get("metrics") or {}
    if "generation_tokens_per_sec" in metrics:
        return float(metrics["generation_tokens_per_sec"])

    n_gen = command_arg(record.get("command", ""), "-n")
    if not n_gen:
        return None

    elapsed = float(record.get("elapsed_sec") or 0)
    if elapsed <= 0:
        return None
    return float(n_gen) / elapsed


def latest_success_by_label(records):
    latest = {}
    for record in records:
        if record.get("returncode") != 0:
            continue
        if record.get("suite") != "cli":
            continue
        label = record.get("label")
        if label:
            latest[label] = record
    return latest


def save_chart(chart, output_dir, stem):
    html_path = output_dir / f"{stem}.html"
    svg_path = output_dir / f"{stem}.svg"
    chart.to_html(html_path)
    chart.to_svg(svg_path)
    return html_path, svg_path


def make_elapsed_chart(records, output_dir):
    labels = list(records)
    elapsed = [float(records[label]["elapsed_sec"]) for label in labels]
    chart = xy.bar_chart(
        xy.bar(labels, elapsed, color="#2563eb", corner_radius=4),
        xy.x_axis(label="Experiment"),
        xy.y_axis(label="Elapsed seconds"),
        THEME,
        title="Benchmark Elapsed Time (seconds, lower is better)",
    )
    return save_chart(chart, output_dir, "elapsed_seconds")


def make_throughput_chart(records, output_dir):
    labels = []
    rates = []
    for label, record in records.items():
        rate = estimate_tokens_per_second(record)
        if rate is None:
            continue
        labels.append(label)
        rates.append(rate)

    chart = xy.bar_chart(
        xy.bar(labels, rates, color="#16a34a", corner_radius=4),
        xy.x_axis(label="Experiment"),
        xy.y_axis(label="Estimated tokens/sec"),
        THEME,
        title="Estimated Generation Throughput (tokens/sec, higher is better)",
    )
    return save_chart(chart, output_dir, "estimated_tokens_per_second")


def make_threads_chart(records, output_dir):
    points = []
    for label, record in records.items():
        match = re.fullmatch(r"threads/(\d+)", label)
        if not match:
            continue
        rate = estimate_tokens_per_second(record)
        if rate is not None:
            points.append((int(match.group(1)), rate))

    points.sort()
    if not points:
        return None

    x_values = [threads for threads, _ in points]
    y_values = [rate for _, rate in points]
    chart = xy.line_chart(
        xy.line(x_values, y_values, color="#7c3aed", width=3),
        xy.scatter(x_values, y_values, color="#7c3aed", size=42),
        xy.x_axis(label="Threads"),
        xy.y_axis(label="Estimated tokens/sec"),
        THEME,
        title="Thread Scaling (threads vs estimated tokens/sec)",
    )
    return save_chart(chart, output_dir, "thread_scaling")


def parse_perf_stat(path):
    text = path.read_text(encoding="utf-8")
    fields = {}

    patterns = {
        "CPUs utilized": r"#\s+([0-9.]+)\s+CPUs utilized",
        "IPC": r"#\s+([0-9.]+)\s+insn per cycle",
        "CPU GHz": r"#\s+([0-9.]+)\s+GHz",
        "L1 miss %": r"#\s+([0-9.]+)% of all L1-dcache accesses",
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            fields[label] = float(match.group(1))

    elapsed_match = re.search(r"([0-9.]+)\s+seconds time elapsed", text)
    if elapsed_match:
        fields["Elapsed sec"] = float(elapsed_match.group(1))

    return fields


def make_perf_chart(perf_path, output_dir):
    if not perf_path.exists():
        return None

    fields = parse_perf_stat(perf_path)
    if not fields:
        return None

    labels = list(fields)
    values = [fields[label] for label in labels]
    chart = xy.bar_chart(
        xy.bar(labels, values, color="#f59e0b", corner_radius=4),
        xy.x_axis(label="perf stat metric"),
        xy.y_axis(label="Value"),
        THEME,
        title="perf stat Summary",
    )
    return save_chart(chart, output_dir, "perf_summary")


def make_speedup_chart(records, output_dir):
    baseline = records.get("kv/q8_0") or records.get("threads/4")
    if baseline is None:
        return None

    baseline_elapsed = float(baseline.get("elapsed_sec") or 0)
    if baseline_elapsed <= 0:
        return None

    labels = []
    speedups = []
    for label, record in records.items():
        elapsed = float(record.get("elapsed_sec") or 0)
        if elapsed <= 0:
            continue
        labels.append(label)
        speedups.append(baseline_elapsed / elapsed)

    chart = xy.bar_chart(
        xy.bar(labels, speedups, color="#0891b2", corner_radius=4),
        xy.x_axis(label="Experiment"),
        xy.y_axis(label="Speedup vs kv/q8_0 baseline"),
        THEME,
        title="Relative Speed (1.0 = q8 KV baseline)",
    )
    return save_chart(chart, output_dir, "relative_speedup")


def write_index(paths, output_dir):
    links = []
    for html_path, svg_path in paths:
        links.append(
            f'<li><a href="{html_path.name}">{html_path.stem}</a> '
            f'(<a href="{svg_path.name}">svg</a>)</li>'
        )

    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pi Oven Benchmark Plots</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 40px; color: #171717; }}
    li {{ margin: 0.5rem 0; }}
  </style>
</head>
<body>
  <h1>Pi Oven Benchmark Plots</h1>
  <ul>
    {"".join(links)}
  </ul>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(index, encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/benchmark_results.jsonl")
    parser.add_argument("--perf", type=Path, default=ROOT / "results/perf_stat.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/plots")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = latest_success_by_label(load_jsonl(args.results))
    if not records:
        raise SystemExit(f"No successful llama-cli rows found in {args.results}")

    outputs = [
        make_elapsed_chart(records, args.output_dir),
        make_throughput_chart(records, args.output_dir),
    ]

    optional_outputs = [
        make_threads_chart(records, args.output_dir),
        make_speedup_chart(records, args.output_dir),
        make_perf_chart(args.perf, args.output_dir),
    ]
    outputs.extend(output for output in optional_outputs if output is not None)

    index = write_index(outputs, args.output_dir)
    print(f"Wrote {index}")
    for html_path, svg_path in outputs:
        print(f"Wrote {html_path}")
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
