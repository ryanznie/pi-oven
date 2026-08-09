# Pi Oven Results Analysis

## Executive Summary

The Raspberry Pi completed all nine runs successfully: seven llama-cli experiments and two llama-bench experiments. Exact throughput comes from llama-bench; CLI records are analyzed only by wall time because their timing text, actual generated-token count, and TTFT were not captured.

- Prefill: F16 KV was fastest at **7.112 tokens/s**; Q8 KV reached 6.811 tokens/s (-4.23%).
- Generation: F16 KV reached **3.080 tokens/s** and Q8 KV reached 3.059 tokens/s (-0.67%). The difference is negligible in this one-run sample.
- Thread scaling: 2 threads reduced wall time from 177.005s to 98.422s (1.80x). Four threads reached 96.317s (1.84x versus one), only 2.2% faster than two threads.
- Speculative decoding took 209.712s versus 96.580s for the Q8 baseline, or **2.17x longer**. Assuming comparable completion lengths, the 0.5B draft overhead did not pay off on this configuration.

## Exact llama-bench Measurements

| workload | kv_cache | tokens | tokens_per_sec | latency_sec | threads |
| --- | --- | --- | --- | --- | --- |
| generation | k=f16, v=f16 | 256 | 3.07956 | 83.128747 | 4 |
| generation | k=q8_0, v=q8_0 | 256 | 3.058964 | 83.688457 | 4 |
| prefill | k=f16, v=f16 | 512 | 7.111643 | 71.994612 | 4 |
| prefill | k=q8_0, v=q8_0 | 512 | 6.81072 | 75.175609 | 4 |

These are the strongest throughput measurements because llama-bench reports the actual workload and timing directly. There was one sample per condition and no warmup, so small differences should not be treated as conclusive.

## llama-cli Wall Time

| label | elapsed_sec | threads | kv_cache | requested_max_tokens |
| --- | --- | --- | --- | --- |
| kv/q4_0 | 14.502 | 4 | k=q4_0, v=q4_0 | 256 |
| threads/4 | 96.317 | 4 | k=q8_0, v=q8_0 | 256 |
| kv/q8_0 | 96.58 | 4 | k=q8_0, v=q8_0 | 256 |
| kv/f16 | 97.229 | 4 | k=f16, v=f16 | 256 |
| threads/2 | 98.422 | 2 | k=q8_0, v=q8_0 | 256 |
| threads/1 | 177.005 | 1 | k=q8_0, v=q8_0 | 256 |
| speculative/draft-simple | 209.712 | 4 | k=q8_0, v=q8_0 | 256 |

The requested maximum is not proof that all 256 tokens were generated: generation can stop at EOS. In particular, `kv/q4_0` completed in 14.502s while comparable runs took about 96-97s, so its apparent speed is almost certainly an early-stop artifact. It is excluded from throughput and speedup claims.

## CPU Profile

- Elapsed time: **49.188s**
- Average CPU use: **3.628 cores** (90.7% of four cores)
- Instructions per cycle: **0.82 IPC**
- Reported frequency: **1.771 GHz**
- L1 data-cache miss rate: **0.69%**
- User/system CPU time: **177.258s / 1.100s**

The workload used most of the four-core CPU budget. IPC of 0.82 indicates substantial pipeline stalls, while the low L1 miss rate suggests the bottleneck is not simply L1 behavior; model compute, wider cache levels, and memory bandwidth remain likely constraints.

## Data Gaps and Next Run

1. Run each llama-bench condition at least three times with warmup and report mean plus standard deviation.
2. Fix CLI output capture so prompt-eval rate, generation rate, actual token count, and TTFT are recorded.
3. Repeat Q4 KV with a prompt or flags that force a fixed output length before comparing it.
4. Record speculative acceptance rate; wall time alone cannot explain whether the draft model proposed useful tokens.
5. Capture process peak RAM and temperature to detect memory pressure or thermal throttling.
