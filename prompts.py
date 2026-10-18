"""Запросы ровно заданной длины (§3.3): метка + инструкция + корпус."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from client import Client

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "pride_and_prejudice.txt"
INSTRUCTION = "Summarize the following text in one sentence."

_corpus_tokens: list[int] | None = None


def _post(server_root: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_root.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def make_messages(
    client: Client,
    n_tokens: int,
    req_id: str,
    *,
    offset: int = 0,
) -> list[dict[str, str]]:
    """Messages с ровно n_tokens после chat-template (метка входит в длину)."""
    global _corpus_tokens
    root = client.server_root
    if _corpus_tokens is None:
        text = CORPUS.read_text(encoding="utf-8")
        _corpus_tokens = _post(root, "/tokenize", {"content": text})["tokens"]

    corpus = _corpus_tokens
    if offset < 0 or offset >= len(corpus):
        raise SystemExit(f"offset={offset} вне корпуса (len={len(corpus)})")

    def build(body_n: int) -> tuple[list[dict[str, str]], int]:
        body_n = max(0, min(body_n, len(corpus) - offset))
        chunk = corpus[offset : offset + body_n]
        body = _post(root, "/detokenize", {"tokens": chunk})["content"] if chunk else ""
        content = f"[{req_id}] {INSTRUCTION}"
        if body:
            content = f"{content}\n\n{body}"
        msgs = [{"role": "user", "content": content}]
        prompt = _post(root, "/apply-template", {"messages": msgs})["prompt"]
        n = len(_post(root, "/tokenize", {"content": prompt})["tokens"])
        return msgs, n

    # сколько токенов уже съедают метка + инструкция + template
    msgs, n = build(0)
    if n > n_tokens:
        raise SystemExit(f"overhead={n} > цели {n_tokens}")
    if n == n_tokens:
        return msgs

    # берём примерно недостающие токены корпуса и доводим по одному
    body_n = min(n_tokens - n, len(corpus) - offset)
    msgs, n = build(body_n)
    while n < n_tokens and body_n < len(corpus) - offset:
        body_n += 1
        msgs, n = build(body_n)
    while n > n_tokens and body_n > 0:
        body_n -= 1
        msgs, n = build(body_n)

    if n != n_tokens:
        raise SystemExit(
            f"Нельзя попасть ровно в {n_tokens} токенов "
            f"(ближайшее: {n} при body_n={body_n}, req_id={req_id!r})"
        )
    return msgs
