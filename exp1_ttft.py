"""Эксп. 1 (§4.1): длина запроса → TTFT; опционально prefix cache.

Сервер: ./scripts/start_server.sh              # без cache, -np 1
        ./scripts/start_server.sh --enable-prefix-cache  # для --prefix
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from client import Client
from measure import timed
from prompts import make_messages
from runlog import append, finish, start

PROMPT_LENS_CPU = [128, 512, 1024, 2048, 4096]
PROMPT_LENS_GPU = [256, 1024, 4096, 8192, 16384]
MAX_TOKENS = 16
REPEATS = 5


def main() -> None:
    p = argparse.ArgumentParser(description="ЛР1 эксп.1: TTFT vs длина запроса")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--prefix", action="store_true", help="п.3: один запрос 3 раза с cache")
    args = p.parse_args()

    c = Client()
    lengths = PROMPT_LENS_GPU if args.gpu else PROMPT_LENS_CPU

    if args.prefix:
        L = 8192 if args.gpu else 2048
        run = start("prefix_cache", {"prompt_tokens": L, "repeats": 3})
        msgs = make_messages(c, L, "prefix")
        ttfts = []
        for i in range(3):
            r = timed(c, msgs, MAX_TOKENS)
            append(
                run,
                {"id": f"prefix-{i}", "prompt_tokens_target": L, **r},
                messages=msgs,
            )
            ttfts.append(r["ttft_s"])
            print(f"  #{i}: ttft={r['ttft_s']:.3f}s")
        speedup = ttfts[0] / ttfts[1] if ttfts[1] > 0 else None
        finish(run, {"series": "prefix_cache", "speedup_1_over_2": speedup, "ttft_s": ttfts})
        return

    run = start(
        "ttft_vs_prompt",
        {"prompt_tokens": lengths, "max_tokens": MAX_TOKENS, "repeats": REPEATS},
    )
    timed(c, make_messages(c, lengths[0], "warmup"), MAX_TOKENS)

    summary = []
    for L in lengths:
        ttfts = []
        for i in range(REPEATS):
            msgs = make_messages(c, L, f"L{L}-{i}")
            r = timed(c, msgs, MAX_TOKENS)
            append(
                run,
                {
                    "id": f"L{L}-{i}",
                    "prompt_tokens_target": L,
                    "ttft_s": r["ttft_s"],
                    "prompt_tokens": r["prompt_tokens"],
                    "output_tokens": r["output_tokens"],
                    "e2e_s": r["e2e_s"],
                    "tpot_s": r["tpot_s"],
                    "finish_reason": r["finish_reason"],
                    "text": r["text"],
                },
                messages=msgs,
            )
            ttfts.append(r["ttft_s"])
            print(f"  L={L} #{i}: ttft={r['ttft_s']:.3f}s prompt={r['prompt_tokens']}")
        med = statistics.median(ttfts)
        summary.append(
            {
                "L": L,
                "ttft_median": med,
                "ttft_min": min(ttfts),
                "ttft_max": max(ttfts),
                "prefill_tok_s": L / med if med else None,
            }
        )

    fig = Path(run) / "figures" / "ttft_vs_prompt.png"
    xs = [s["L"] for s in summary]
    ys = [s["ttft_median"] for s in summary]
    yerr = [
        [s["ttft_median"] - s["ttft_min"] for s in summary],
        [s["ttft_max"] - s["ttft_median"] for s in summary],
    ]
    plt.figure(figsize=(7, 4))
    plt.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=4)
    plt.xlabel("Длина запроса, токены")
    plt.ylabel("TTFT, с")
    plt.title("Эксп. 1: TTFT от длины запроса")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig, dpi=150)
    plt.close()

    finish(run, {"series": "ttft_vs_prompt", "by_length": summary}, figures=[fig])


if __name__ == "__main__":
    main()
