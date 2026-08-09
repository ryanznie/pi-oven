# TinyServe

TinyServe is the learning track in this repository. It uses Hugging Face only
to load/tokenize a model and execute PyTorch forward passes. Its decoding and
cache logic lives in this directory. It never calls `model.generate()`.

Only Phase 1 is implemented: deterministic, naive autoregressive decoding.
There is deliberately no KV cache yet.

## What Phase 1 Does

For every generated token, `generate_naive()` sends the complete growing token
sequence through the model:

```text
prompt length = 10
forward input lengths = [10, 11, 12, 13]
```

The model computes logits for every position, but we use only the last one to
select the next token. This repeated work is the baseline that Phase 2 will
optimize with `past_key_values`.

## Install on Raspberry Pi

From the repository root:

```bash
python3 -m venv .venv-tinyserve
.venv-tinyserve/bin/python -m pip install --upgrade pip
.venv-tinyserve/bin/python -m pip install -r tinyserve/requirements.txt
```

The first run downloads model weights from Hugging Face, so the Pi needs an
internet connection for that run. Later runs use the local Hugging Face cache.

## Run Phase 1

Start with four generated tokens so you can verify the loop quickly:

```bash
.venv-tinyserve/bin/python -m tinyserve.benchmark --max-new-tokens 4 --threads 4
```

Then run a longer measurement:

```bash
.venv-tinyserve/bin/python -m tinyserve.benchmark --max-new-tokens 16 --threads 4
```

The default is `HuggingFaceTB/SmolLM2-135M`, chosen because naive decoding is
intentionally slow and repeatedly processes the prompt. Model selection stays
configurable:

```bash
.venv-tinyserve/bin/python -m tinyserve.benchmark \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 4 \
  --threads 4
```

Qwen 0.5B uses substantially more memory and compute than the default model.

## Expected Output

Exact timings and generated text vary, but the important pattern is:

```text
Naive decoding (KV cache disabled):
  forward 01: input shape=(1, 12), tokens processed=12
  forward 02: input shape=(1, 13), tokens processed=13
  forward 03: input shape=(1, 14), tokens processed=14
  forward 04: input shape=(1, 15), tokens processed=15

Result
  generated tokens: 4
  forward input lengths: [12, 13, 14, 15]
  total input tokens processed: 54
```

Results append to `results/tinyserve_phase1.csv` with fields that can later be
normalized alongside llama.cpp results. A fair runtime comparison must use the
same model, prompt, generated-token count, thread count, and clearly report
weight precision. The existing Qwen 1.5B Q4 llama.cpp results are useful system
measurements, but are not directly comparable to SmolLM2-135M float32.

## Profile on Raspberry Pi

```bash
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses \
  .venv-tinyserve/bin/python -m tinyserve.benchmark --max-new-tokens 8

perf record -g -- \
  .venv-tinyserve/bin/python -m tinyserve.benchmark --max-new-tokens 8

perf report
```

Use `task-clock` for CPU time, cycles and instructions for work performed, IPC
(`instructions / cycles`) for pipeline efficiency, and cache misses for memory
behavior. Model loading is included by these commands; a later benchmark phase
will separate cold-start and steady-state measurements.

## Important Code

- `models.py` loads the model and tokenizer.
- `naive.py` contains the generation loop worth reading line by line.
- `benchmark.py` runs the loop, prints metrics, and appends one CSV row.

Future files for KV caching, prefix caching, and speculative decoding should be
added one phase at a time after this baseline runs successfully on the Pi.
