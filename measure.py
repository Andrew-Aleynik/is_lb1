from __future__ import annotations

import time
from typing import Any

from client import Client


def _delta_text(delta: Any) -> str | None:
    for key in ("content", "reasoning_content", "reasoning"):
        val = getattr(delta, key, None)
        if val:
            return val
    return None


def timed(
    client: Client,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    temperature: float = 0,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    stream = client.stream(
        messages, max_tokens, temperature=temperature, ignore_eos=ignore_eos
    )
    t1 = t_last = None
    usage, parts = None, []
    finish = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            finish = ch.finish_reason
        piece = _delta_text(ch.delta)
        if piece:
            now = time.perf_counter()
            t1 = t1 or now
            t_last = now
            parts.append(piece)
    if usage is None or t1 is None or t_last is None:
        raise RuntimeError("пустой ответ или нет usage")
    n = usage.completion_tokens
    return {
        "ttft_s": t1 - t0,
        "tpot_s": (t_last - t1) / (n - 1) if n > 1 else None,
        "e2e_s": t_last - t0,
        "prompt_tokens": usage.prompt_tokens,
        "output_tokens": n,
        "finish_reason": finish,
        "text": "".join(parts),
    }
