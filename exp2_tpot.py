"""Эксп. 2 (§4.2): длина ответа → e2e / TPOT.

Сервер: ./scripts/start_server.sh   # -np 1, без prefix cache
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from client import Client
from measure import timed
from prompts import make_messages
from runlog import append, finish, start

PROMPT_TOKENS = 1024
OUTPUT_LENS = [16, 64, 256, 1024]
REPEATS = 5


def main() -> None:
    c = Client()
    run = start(
        "tpot_vs_output",
        {"prompt_tokens": PROMPT_TOKENS, "max_tokens": OUTPUT_LENS, "repeats": REPEATS},
    )
    timed(c, make_messages(c, PROMPT_TOKENS, "warmup"), 16)

    summary = []
    for out_n in OUTPUT_LENS:
        e2es, tpots = [], []
        for i in range(REPEATS):
            msgs = make_messages(c, PROMPT_TOKENS, f"O{out_n}-{i}")
            r = timed(c, msgs, out_n)
            append(
                run,
                {
                    "id": f"O{out_n}-{i}",
                    "max_tokens": out_n,
                    "output_tokens": r["output_tokens"],
                    "e2e_s": r["e2e_s"],
                    "tpot_s": r["tpot_s"],
                    "ttft_s": r["ttft_s"],
                    "prompt_tokens": r["prompt_tokens"],
                    "finish_reason": r["finish_reason"],
                    "text": r["text"],
                },
                messages=msgs,
            )
            e2es.append(r["e2e_s"])
            if r["tpot_s"] is not None:
                tpots.append(r["tpot_s"])
            print(f"  out={out_n} #{i}: e2e={r['e2e_s']:.3f}s tpot={r['tpot_s']}")
        summary.append(
            {
                "N": out_n,
                "e2e_median": statistics.median(e2es),
                "tpot_median": statistics.median(tpots) if tpots else None,
            }
        )

    xs = [s["N"] for s in summary]
    ys = [s["e2e_median"] for s in summary]
    slope, intercept = np.polyfit(xs, ys, 1)

    fig = Path(run) / "figures" / "e2e_vs_output.png"
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, "o", label="медиана e2e")
    xline = np.linspace(min(xs), max(xs), 50)
    plt.plot(xline, intercept + slope * xline, "-", label=f"≈ {intercept:.3f}+{slope:.4f}·N")
    plt.xlabel("Длина ответа, токены")
    plt.ylabel("Полное время, с")
    plt.title("Эксп. 2: e2e от длины ответа (наклон ≈ TPOT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig, dpi=150)
    plt.close()

    finish(
        run,
        {
            "series": "tpot_vs_output",
            "fit_intercept_s": float(intercept),
            "fit_slope_s": float(slope),
            "by_output": summary,
        },
        figures=[fig],
    )


if __name__ == "__main__":
    main()
