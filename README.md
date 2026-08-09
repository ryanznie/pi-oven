# pi-oven

Small Raspberry Pi 5 project for benchmarking local LLM inference with
`llama.cpp`, GGUF Qwen models, KV-cache settings, speculative decoding, and
Linux `perf`.

The repo now has two tracks:

- `llama.cpp` benchmarking: measure existing optimized inference features.
- `TinyServe`: implement small Python + PyTorch inference loops ourselves for
  learning.

This intentionally avoids 3B models. The default target is
`Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`; the draft model for speculative decoding is
`Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`.

## Raspberry Pi Setup

On a fresh Raspberry Pi OS / Debian ARM64 install:

```bash
git clone <this-repo-url> pi-oven
cd pi-oven
./scripts/setup_pi.sh
```

That installs build tools, clones `llama.cpp` into `./vendor/llama.cpp`, and
builds the CLI tools. The script tries the Arm KleidiAI CPU backend first, then
falls back to a plain CPU build if needed.

Download the two small models:

```bash
./scripts/download_models.sh
```

Run a smoke test:

```bash
./scripts/smoke_test.sh
```

## TinyServe Learning Engine

TinyServe is the from-scratch learning path. Phase 1 contains a visible,
greedy autoregressive loop that intentionally disables the KV cache:

```bash
python3 -m venv .venv-tinyserve
.venv-tinyserve/bin/python -m pip install -r tinyserve/requirements.txt
.venv-tinyserve/bin/python -m tinyserve.benchmark --max-new-tokens 4 --threads 4
```

See [`tinyserve/README.md`](tinyserve/README.md) for the concept, expected
output, a longer run, and Raspberry Pi `perf` commands. TinyServe records a
comparison-friendly CSV now; a unified same-model comparison with llama.cpp
will be added after the learning implementations are in place.

## Run Benchmarks

Run the full long benchmark suite:

```bash
python3 benchmarks/run_benchmarks.py
```

By default this uses `512` prompt tokens, `256` generated tokens, runs KV-cache
tests, thread tests, speculative decoding, then `llama-bench` last. Regular
`llama-cli` runs get up to `2400` seconds each; `llama-bench` gets `300`
seconds each by default.

Results are written to `results/benchmark_results.jsonl`.

Useful variants:

```bash
# Fast debug run. This skips llama-bench and speculative decoding.
python3 benchmarks/run_benchmarks.py --quick --skip-speculative

# Use a different llama.cpp build or model directory.
python3 benchmarks/run_benchmarks.py \
  --llama-bin ./vendor/llama.cpp/build/bin \
  --model-dir ./models

# Run only KV-cache experiments.
python3 benchmarks/run_benchmarks.py --suite kv

# Run llama-bench explicitly. This can be slow on Raspberry Pi.
python3 benchmarks/run_benchmarks.py --suite bench --timeout 300

# Run speculative decoding experiments too.
python3 benchmarks/run_benchmarks.py --suite speculative
```

The script prints each experiment before it starts. On a Raspberry Pi 4, the
default full run can take a long time. `llama-bench` runs last and streams its
own output live; if it still stalls, the earlier `llama-cli` results are already
saved. Quick mode keeps generated tokens low and skips `llama-bench` by default
unless `--include-bench` or `--suite bench` is used.

## CPU Profiling

Use `perf stat` first:

```bash
./scripts/perf_stat.sh
```

For a sampled profile:

```bash
./scripts/perf_record.sh
perf report
```

## Plot Results

Install the plotting dependency and render the benchmark plots:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/plot_results.py
```

The script uses `xy` and writes interactive HTML plus SVG exports to
`results/plots/`.

Generate a CSV and markdown analytics report:

```bash
python3 scripts/analyze_results.py
```

The report is written to `results/analytics/report.md`.

Useful ordering options:

```bash
python3 scripts/analyze_results.py --sort-by tps
python3 scripts/analyze_results.py --sort-by elapsed
python3 scripts/analyze_results.py --sort-by ttft
```

New benchmark runs also record TTFT, measured as wall-clock time from process
start to the first generated stdout token.

If the Pi denies access to hardware counters, temporarily relax perf paranoia:

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

## What To Measure

Start with these comparisons:

1. Prompt processing speed via `llama-bench`.
2. Token generation speed via `llama-bench`.
3. KV cache type: `f16`, `q8_0`, `q4_0`.
4. Thread count: `1`, `2`, `4`.
5. Speculative decoding with the 0.5B draft model.
6. CPU counters from `perf stat`: cycles, instructions, cache misses, branches.

The important output is tokens/sec and whether each optimization actually helps
on Raspberry Pi 5. Some settings that help on a laptop may do little or even
hurt on the Pi, which is exactly the experiment.

## Paths

Defaults used by the scripts:

- llama.cpp: `./vendor/llama.cpp`
- llama.cpp binaries: `./vendor/llama.cpp/build/bin`
- target model: `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- draft model: `./models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
- results: `./results`

## Model Sources

- Target model:
  [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- Draft model:
  [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
- Runtime:
  [llama.cpp](https://github.com/ggml-org/llama.cpp)
