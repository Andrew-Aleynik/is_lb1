"""Эксп. 3 (§4.3): нагрузка → throughput / TTFT / TPOT / goodput.

Сервер: ./scripts/start_server.sh --np 4
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from client import Client
from measure import timed
from prompts import make_messages
from runlog import append, finish, start

PROMPT_TOKENS = 1024
MAX_TOKENS = 128
LEVELS_CPU = [1, 2, 4]
LEVELS_GPU = [1, 2, 4, 8, 16, 32]
SLO_CPU = {"ttft_p90_s": 15.0, "tpot_median_s": 0.20}
SLO_GPU = {"ttft_p90_s": 2.0, "tpot_median_s": 0.10}


def run_level(concurrency: int, n_req: int) -> tuple[list[dict], float]:
    jobs: Queue = Queue()
    for i in range(n_req):
        jobs.put((f"c{concurrency}-{i}", i))
    for _ in range(concurrency):
        jobs.put(None)

    rows: list[dict] = []

    def worker() -> list[dict]:
        c = Client()
        out: list[dict] = []
        while True:
            item = jobs.get()
            if item is None:
                return out
            rid, idx = item
            msgs = make_messages(c, PROMPT_TOKENS, rid, offset=idx * 64)
            t_send = time.perf_counter()
            r = timed(c, msgs, MAX_TOKENS)
            t_done = time.perf_counter()
            out.append(
                {
                    "id": rid,
                    "concurrency": concurrency,
                    "max_tokens": MAX_TOKENS,
                    "prompt_tokens_target": PROMPT_TOKENS,
                    "t_send": t_send,
                    "t_done": t_done,
                    "messages": msgs,
                    **r,
                }
            )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(concurrency)]
        for f in as_completed(futs):
            rows.extend(f.result())

    wall = max(r["t_done"] for r in rows) - min(r["t_send"] for r in rows)
    return rows, wall


def main() -> None:
    p = argparse.ArgumentParser(description="ЛР1 эксп.3: нагрузка")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    levels = LEVELS_GPU if args.gpu else LEVELS_CPU
    slo = SLO_GPU if args.gpu else SLO_CPU
    c = Client()
    run = start(
        "load",
        {"levels": levels, "prompt_tokens": PROMPT_TOKENS, "max_tokens": MAX_TOKENS, "slo": slo},
    )
    timed(c, make_messages(c, PROMPT_TOKENS, "warmup"), 16)

    table = []
    for conc in levels:
        n_req = max(16, 4 * conc)
        print(f"c={conc}, n={n_req}")
        rows, wall = run_level(conc, n_req)
        for row in rows:
            msgs = row.pop("messages", None)
            append(run, row, messages=msgs)
        ttfts = [r["ttft_s"] for r in rows]
        tpots = [r["tpot_s"] for r in rows if r["tpot_s"] is not None]
        thr = sum(r["output_tokens"] for r in rows) / wall
        ttft_p90 = float(np.percentile(ttfts, 90))
        tpot_med = statistics.median(tpots)
        good = [
            r
            for r in rows
            if r["ttft_s"] <= slo["ttft_p90_s"]
            and (r["tpot_s"] or 0) <= slo["tpot_median_s"]
        ]
        goodput = len(good) / wall
        ok = ttft_p90 <= slo["ttft_p90_s"] and tpot_med <= slo["tpot_median_s"]
        row = {
            "c": conc,
            "throughput_tok_s": thr,
            "rps": len(rows) / wall,
            "ttft_p90": ttft_p90,
            "tpot_median": tpot_med,
            "goodput_rps": goodput,
            "within_slo": ok,
        }
        table.append(row)
        print(
            f"  thr={thr:.2f} tok/s  ttft_p90={ttft_p90:.3f}s  "
            f"tpot={tpot_med:.4f}s  goodput={goodput:.3f}/s  slo={ok}"
        )

    fig = Path(run) / "figures" / "throughput_vs_latency.png"
    xs = [r["throughput_tok_s"] for r in table]
    ys = [r["ttft_p90"] for r in table]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, "o-")
    for r in table:
        plt.annotate(f"c={r['c']}", (r["throughput_tok_s"], r["ttft_p90"]), fontsize=8)
    plt.axhline(slo["ttft_p90_s"], ls="--", color="C1", label="SLO TTFT p90")
    plt.xlabel("Пропускная способность, токены/с")
    plt.ylabel("TTFT p90, с")
    plt.title("Эксп. 3: пропускная способность — задержка")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig, dpi=150)
    plt.close()

    max_ok = max((r["c"] for r in table if r["within_slo"]), default=None)
    finish(
        run,
        {"series": "load", "max_c_within_slo": max_ok, "levels": table},
        figures=[fig],
    )


if __name__ == "__main__":
    main()
