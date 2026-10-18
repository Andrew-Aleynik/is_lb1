#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8080
PID_FILE="$ROOT/logs/llama_server.pid"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--port PORT]"; exit 0 ;;
    *) echo "Неизвестный аргумент: $1" >&2; exit 1 ;;
  esac
done

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE" || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
fi

fuser -k "${PORT}/tcp" 2>/dev/null || true
pkill -f "[l]lama-server.*--port ${PORT}" 2>/dev/null || true
sleep 1
rm -f "$PID_FILE"
echo "stopped :${PORT}"
