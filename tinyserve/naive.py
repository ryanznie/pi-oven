"""Phase 1: naive autoregressive decoding without a KV cache."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float
    total_latency_sec: float
    tokens_per_sec: float
    forward_input_tokens: list[int]
    total_input_tokens_processed: int


def select_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Select the most likely token, with no sampling or randomness.

    logits shape: [batch_size, vocabulary_size]
    return shape: [batch_size, 1]
    """

    return torch.argmax(logits, dim=-1, keepdim=True)


@torch.inference_mode()
def generate_naive(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    device: str = "cpu",
    show_steps: bool = True,
) -> tuple[str, GenerationStats]:
    """Generate text by reprocessing the entire sequence for every token.

    If the prompt has P tokens, the forward passes receive P, P+1, P+2, ...
    tokens. Phase 2 will replace that repeated work with `past_key_values`.
    """

    encoded = tokenizer(prompt, return_tensors="pt")

    # input_ids shape: [batch_size=1, sequence_length]
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_tokens = int(input_ids.shape[1])
    forward_input_tokens: list[int] = []
    generated_token_ids: list[int] = []
    ttft_sec = 0.0
    started = time.perf_counter()

    for step in range(max_new_tokens):
        current_length = int(input_ids.shape[1])
        forward_input_tokens.append(current_length)
        if show_steps:
            print(
                f"  forward {step + 1:02d}: input shape={tuple(input_ids.shape)}, "
                f"tokens processed={current_length}",
                flush=True,
            )

        # use_cache=False is the defining choice in this phase. The model receives
        # [1, current_length] and recomputes attention for every previous token.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # outputs.logits shape: [1, current_length, vocabulary_size]. Only the
        # final position predicts the next token, so select [:, -1, :].
        next_token_logits = outputs.logits[:, -1, :]
        next_token = select_greedy(next_token_logits)

        if step == 0:
            ttft_sec = time.perf_counter() - started

        token_id = int(next_token.item())
        generated_token_ids.append(token_id)

        # Append the selected token. The next iteration sends this whole growing
        # tensor through the model again.
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    total_latency_sec = time.perf_counter() - started
    generated_tokens = len(generated_token_ids)
    tokens_per_sec = (
        generated_tokens / total_latency_sec if total_latency_sec > 0 else 0.0
    )
    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    stats = GenerationStats(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_sec=ttft_sec,
        total_latency_sec=total_latency_sec,
        tokens_per_sec=tokens_per_sec,
        forward_input_tokens=forward_input_tokens,
        total_input_tokens_processed=sum(forward_input_tokens),
    )
    return generated_text, stats
