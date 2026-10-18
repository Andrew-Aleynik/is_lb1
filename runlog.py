from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import openai
import yaml
from mlflow.entities import SpanType

ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def _safe_id(req_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in req_id) or "anon"


def _read_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def start(series: str, config: dict[str, Any]) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"{run_id}_{series}"
    run_dir.mkdir(parents=True)
    (run_dir / "figures").mkdir()
    (run_dir / "traces").mkdir()

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    endpoint: dict[str, Any] = {}
    ep_path = ROOT / "endpoint.yaml"
    if ep_path.is_file():
        with ep_path.open(encoding="utf-8") as f:
            endpoint = yaml.safe_load(f) or {}

    corpus = ROOT / "data" / "pride_and_prejudice.txt"
    manifest = ROOT / "data" / "manifest.json"
    meta = {
        "run_id": run_dir.name,
        "lab": "LB1",
        "series": series,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "openai": getattr(openai, "__version__", "unknown"),
            "mlflow": getattr(mlflow, "__version__", "unknown"),
        },
        "endpoint": {
            "base_url": endpoint.get("base_url"),
            "model": endpoint.get("model"),
        },
        "data_sha256": {
            "pride_and_prejudice.txt": _sha256(corpus) if corpus.is_file() else None,
            "manifest.json": _sha256(manifest) if manifest.is_file() else None,
        },
        "error": None,
        "mlflow_run_id": None,
    }
    _write_meta(run_dir, meta)
    (run_dir / "results.jsonl").touch()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment("LB1")
    active = mlflow.start_run(run_name=run_dir.name)
    meta["mlflow_run_id"] = active.info.run_id
    _write_meta(run_dir, meta)
    mlflow.set_tags(
        {
            "lab": "LB1",
            "series": series,
            "commit": meta["commit"] or "",
            "model": str(endpoint.get("model") or ""),
        }
    )
    flat_params = {
        k: v
        for k, v in config.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
    if flat_params:
        mlflow.log_params({k: str(v) for k, v in flat_params.items()})

    return run_dir


def append(
    run_dir: Path,
    row: dict[str, Any],
    *,
    messages: list[dict[str, str]] | None = None,
) -> None:
    """results.jsonl + traces/<id>.json (сообщения, ответ, время, токены)."""
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if messages is None and "text" not in row:
        return

    req_id = str(row.get("id", "anon"))
    trace = {
        "id": req_id,
        "messages": messages,
        "response": row.get("text"),
        "status": row.get("status", "ok"),
        "error": row.get("error"),
        "ttft_s": row.get("ttft_s"),
        "tpot_s": row.get("tpot_s"),
        "e2e_s": row.get("e2e_s"),
        "prompt_tokens": row.get("prompt_tokens"),
        "output_tokens": row.get("output_tokens"),
        "finish_reason": row.get("finish_reason"),
        "condition": {
            k: row[k]
            for k in (
                "prompt_tokens_target",
                "max_tokens",
                "concurrency",
                "warmup",
            )
            if k in row
        }
        or row.get("condition"),
    }
    trace_path = run_dir / "traces" / f"{_safe_id(req_id)}.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with mlflow.start_span(name=req_id, span_type=SpanType.LLM) as span:
        span.set_inputs({"messages": messages})
        span.set_outputs(
            {
                "response": row.get("text"),
                "prompt_tokens": row.get("prompt_tokens"),
                "output_tokens": row.get("output_tokens"),
                "ttft_s": row.get("ttft_s"),
                "tpot_s": row.get("tpot_s"),
                "e2e_s": row.get("e2e_s"),
                "finish_reason": row.get("finish_reason"),
            }
        )


def finish(
    run_dir: Path,
    metrics: dict[str, Any],
    figures: list[Path] | None = None,
    *,
    status: str = "completed",
    error: dict[str, str] | None = None,
) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    meta = _read_meta(run_dir)
    meta["status"] = status
    meta["ended_at"] = _utc_now()
    meta["error"] = error
    _write_meta(run_dir, meta)

    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(k, float(v))
    mlflow.set_tag("status", status)
    mlflow.log_artifacts(str(run_dir))
    for fig in figures or []:
        if fig.is_file():
            mlflow.log_artifact(str(fig), artifact_path="figures")
    mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")
    print(f"saved {run_dir}")
