#!/usr/bin/env bash
# По умолчанию prefix/prompt cache выключен. Включить: --enable-prefix-cache
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/llama-server"
MODEL="$ROOT/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/llama_server.log"
HOST=127.0.0.1
PORT=8080
CTX=16384
NP=1
THREADS=4
ALIAS="Qwen/Qwen3-4B-Instruct-2507"
CACHE_RAM=0

usage() {
  echo "Usage: $0 [--enable-prefix-cache] [--np N] [--port PORT]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-prefix-cache) CACHE_RAM=8192; shift ;;
    --np) NP="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Неизвестный аргумент: $1" >&2; usage ;;
  esac
done

mkdir -p "$LOG_DIR" "$ROOT/models"

if [[ ! -x "$BIN" ]]; then
  echo "Нет $BIN — скачайте llama.cpp ubuntu-x64 (b11714) в bin/" >&2
  exit 1
fi
if [[ ! -e "$MODEL" ]]; then
  BLOB=/usr/share/ollama/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f
  [[ -f "$BLOB" ]] && ln -sfn "$BLOB" "$MODEL" || {
    echo "Нет модели: $MODEL" >&2
    exit 1
  }
fi

"$ROOT/scripts/stop_server.sh" --port "$PORT" >/dev/null

: >"$LOG"
nohup "$BIN" \
  -m "$MODEL" -c "$CTX" -np "$NP" -t "$THREADS" \
  --host "$HOST" --port "$PORT" --alias "$ALIAS" \
  --cache-ram "$CACHE_RAM" --jinja \
  >>"$LOG" 2>&1 &
echo $! >"$LOG_DIR/llama_server.pid"
pid=$(cat "$LOG_DIR/llama_server.pid")

for _ in $(seq 1 120); do
  kill -0 "$pid" 2>/dev/null || {
    echo "Сервер упал, см. $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
  }
  curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null && break
  sleep 1
done

curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null || {
  echo "Не отвечает :${PORT}, см. $LOG" >&2
  exit 1
}

echo "OK http://${HOST}:${PORT}/v1  model=${ALIAS}  np=${NP}  cache-ram=${CACHE_RAM}"
echo "лог: $LOG"
