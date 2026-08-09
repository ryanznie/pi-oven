#!/usr/bin/env python3
"""Render accuracy-first benchmark plots with Reflex XY."""

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
    palette=["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"],
)


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_cli(records):
    latest = {}
    for record in records:
        if record.get("suite") == "cli" and record.get("returncode") == 0:
            latest[record["label"]] = record
    return latest


def extract_bench_rows(records):
    decoder = json.JSONDecoder()
    rows = []
    for record in records:
        if record.get("suite") != "llama-bench" or record.get("returncode") != 0:
            continue
        raw = record.get("raw_output", "")
        for match in re.finditer(r'\{\s*"build_commit"', raw):
            try:
                result, _ = decoder.raw_decode(raw[match.start() :])
            except json.JSONDecodeError:
                continue
            workload = "prefill" if int(result.get("n_prompt", 0)) else "generation"
            rows.append(
                {
                    "workload": workload,
                    "kv": str(result["type_k"]),
                    "tokens_per_sec": float(result["avg_ts"]),
                }
            )
    return rows


def save_chart(chart, output_dir, stem):
    html_path = output_dir / f"{stem}.html"
    svg_path = output_dir / f"{stem}.svg"
    chart.to_html(html_path)
    chart.to_svg(svg_path)
    return html_path, svg_path


def make_bench_chart(rows, workload, output_dir):
    selected = [row for row in rows if row["workload"] == workload]
    selected.sort(key=lambda row: row["tokens_per_sec"], reverse=True)
    labels = [row["kv"] for row in selected]
    values = [row["tokens_per_sec"] for row in selected]
    title_word = "Prefill" if workload == "prefill" else "Generation"
    chart = xy.bar_chart(
        xy.bar(labels, values, color="#16a34a", corner_radius=4),
        xy.x_axis(label="KV cache data type"),
        xy.y_axis(label="Tokens per second"),
        THEME,
        title=f"Exact llama-bench {title_word} Throughput (higher is better)",
    )
    return save_chart(chart, output_dir, f"llama_bench_{workload}_throughput")


def make_cli_elapsed_chart(records, output_dir):
    labels = list(records)
    values = [float(records[label]["elapsed_sec"]) for label in labels]
    chart = xy.bar_chart(
        xy.bar(labels, values, color="#2563eb", corner_radius=4),
        xy.x_axis(label="Experiment"),
        xy.y_axis(label="Wall time (seconds)"),
        THEME,
        title="llama-cli Capped-Run Wall Time (actual token count unavailable)",
    )
    return save_chart(chart, output_dir, "cli_wall_time")


def make_threads_chart(records, output_dir):
    points = []
    for label, record in records.items():
        match = re.fullmatch(r"threads/(\d+)", label)
        if match:
            points.append((int(match.group(1)), float(record["elapsed_sec"])))
    points.sort()
    chart = xy.line_chart(
        xy.line([x for x, _ in points], [y for _, y in points], color="#7c3aed", width=3),
        xy.scatter([x for x, _ in points], [y for _, y in points], color="#7c3aed", size=42),
        xy.x_axis(label="CPU threads"),
        xy.y_axis(label="Wall time (seconds)"),
        THEME,
        title="Thread Scaling Wall Time (lower is better)",
    )
    return save_chart(chart, output_dir, "thread_scaling_wall_time")


def make_speculative_chart(records, output_dir):
    baseline = records.get("kv/q8_0")
    speculative = records.get("speculative/draft-simple")
    if not baseline or not speculative:
        return None
    chart = xy.bar_chart(
        xy.bar(
            ["Q8 baseline", "Speculative"],
            [float(baseline["elapsed_sec"]), float(speculative["elapsed_sec"])],
            color="#dc2626",
            corner_radius=4,
        ),
        xy.x_axis(label="Decoding method"),
        xy.y_axis(label="Wall time (seconds)"),
        THEME,
        title="Speculative Decoding Wall Time (lower is better)",
    )
    return save_chart(chart, output_dir, "speculative_wall_time")


def parse_perf_stat(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    patterns = {
        "CPU cores used": r"#\s+([0-9.]+)\s+CPUs utilized",
        "IPC": r"#\s+([0-9.]+)\s+insn per cycle",
        "CPU GHz": r"#\s+([0-9.]+)\s+GHz",
        "L1 miss %": r"#\s+([0-9.]+)% of all L1-dcache accesses",
    }
    fields = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            fields[label] = float(match.group(1))
    return fields


def make_perf_chart(path, output_dir):
    fields = parse_perf_stat(path)
    if not fields:
        return None
    chart = xy.bar_chart(
        xy.bar(list(fields), list(fields.values()), color="#f59e0b", corner_radius=4),
        xy.x_axis(label="perf stat metric"),
        xy.y_axis(label="Reported value"),
        THEME,
        title="CPU Profile Summary (metrics use different units)",
    )
    return save_chart(chart, output_dir, "perf_summary")


def write_index(paths, output_dir):
    links = "".join(
        f'<li><a href="{html.name}">{html.stem}</a> (<a href="{svg.name}">SVG</a>)</li>'
        for html, svg in paths
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pi Oven Benchmark Analysis</title>
  <style>
    body {{ max-width: 760px; margin: 48px auto; padding: 0 20px; font-family: system-ui, sans-serif; color: #171717; }}
    li {{ margin: 0.7rem 0; }}
    .note {{ color: #57534e; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Pi Oven Benchmark Analysis</h1>
  <p class="note">Throughput charts use exact llama-bench measurements. CLI charts show wall time only because actual generated-token counts and TTFT were not captured.</p>
  <ul>{links}</ul>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
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
    records = load_jsonl(args.results)
    cli = latest_cli(records)
    bench = extract_bench_rows(records)
    if not cli or not bench:
        raise SystemExit("Need successful llama-cli and parseable llama-bench results")

    outputs = [
        make_bench_chart(bench, "prefill", args.output_dir),
        make_bench_chart(bench, "generation", args.output_dir),
        make_cli_elapsed_chart(cli, args.output_dir),
        make_threads_chart(cli, args.output_dir),
    ]
    outputs.extend(
        output
        for output in [
            make_speculative_chart(cli, args.output_dir),
            make_perf_chart(args.perf, args.output_dir),
        ]
        if output is not None
    )
    index = write_index(outputs, args.output_dir)
    print(f"Wrote {index}")
    for html, svg in outputs:
        print(f"Wrote {html}")
        print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
