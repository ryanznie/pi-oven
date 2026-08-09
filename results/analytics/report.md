# Pi Oven Results Analysis

## At-a-Glance Winners

| Decision | Winner | Evidence | Confidence |
| --- | --- | --- | --- |
| Highest measured generation throughput | F16 KV cache | 3.080 vs 3.059 tokens/s for Q8 | Low: one sample |
| Highest measured prefill throughput | F16 KV cache | 7.112 vs 6.811 tokens/s for Q8 | Low: one sample |
| Fastest thread setting | 4 threads | 96.317s wall time | Medium |
| Best thread efficiency | 2 threads | 1.80x faster than one; four adds only 2.2% | Medium |
| Best decoding method | Standard Q8 decoding | 96.580s vs 209.712s speculative | Medium |
| Lowest-memory KV option | Not measured | Q8 should use less cache memory, but RAM was not captured | None |
| Q4 KV result | No conclusion | 14.502s is inconsistent and likely ended early | None |

**Practical choice from this run:** use standard decoding with 4 threads for the lowest measured latency, or 2 threads when you want nearly the same latency with better CPU efficiency. F16 and Q8 KV are tied for generation speed in practice; choose Q8 when memory pressure matters, then measure RAM to confirm the benefit.

## Ranked Method Comparison

| category | rank | method | metric | value | relative_to_best | confidence | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KV cache generation | 1 | k=f16, v=f16 | tokens_per_sec | 3.07956 | 1.0 | low | best measured |
| KV cache generation | 2 | k=q8_0, v=q8_0 | tokens_per_sec | 3.058964 | 0.9933 | low | effectively tied |
| KV cache prefill | 1 | k=f16, v=f16 | tokens_per_sec | 7.111643 | 1.0 | low | best measured |
| KV cache prefill | 2 | k=q8_0, v=q8_0 | tokens_per_sec | 6.81072 | 0.9577 | low | slower |
| CPU threads | 1 | 4 thread(s) | wall_time_sec | 96.317 | 1.0 | medium | fastest; 1.84x vs 1 thread |
| CPU threads | 2 | 2 thread(s) | wall_time_sec | 98.422 | 0.9786 | medium | best efficiency; 1.80x vs 1 thread |
| CPU threads | 3 | 1 thread(s) | wall_time_sec | 177.005 | 0.5441 | medium | slowest; 1.00x vs 1 thread |
| Decoding method | 1 | standard decoding | wall_time_sec | 96.58 | 1.0 | medium | winner |
| Decoding method | 2 | speculative decoding | wall_time_sec | 209.712 | 0.4605 | medium | 2.17x slower |

`relative_to_best` is normalized within each category: 1.0 is the winner. It must not be compared across categories because tokens/sec and wall time are different measurements.

## Detailed Findings

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
