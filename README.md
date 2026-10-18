# ЛР1. Инференс и развёртывание LLM

Профиль **CPU**: `llama-server` (llama.cpp) + GGUF Q4_K_M, контекст 16 384.  
Адрес API: `http://127.0.0.1:8080/v1` — см. `endpoint.yaml`, паспорт железа/ПО — `passport.md`.

| Файл | Назначение |
|------|------------|
| `client.py` | `Client.stream` — вызов API |
| `prompts.py` | запрос ровно на L токенов (`make_messages`) |
| `measure.py` | TTFT / TPOT / e2e |
| `exp1_ttft.py` | эксп. 1: TTFT vs длина; `--prefix` — п.3 |
| `exp2_tpot.py` | эксп. 2: e2e / TPOT vs длина ответа |
| `exp3_load.py` | эксп. 3: нагрузка |
| `runlog.py` | `runs/` + traces + MLflow |
| `forecast.md` | прогноз до экспериментов |
| `data/` | Pride and Prejudice + `manifest.json` |
| `scripts/` | `start_server.sh` / `stop_server.sh` |

## Сервер

Нужны `bin/llama-server` и `models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

```bash
./scripts/start_server.sh                         # эксп. 1–2: -np 1, без prefix cache
./scripts/start_server.sh --enable-prefix-cache   # эксп. 1 п.3
./scripts/start_server.sh --np 4                  # эксп. 3
./scripts/stop_server.sh
```

## Клиент и эксперименты

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python client.py                        # smoke

# эксп. 1 (сервер без cache, -np 1)
.venv/bin/python exp1_ttft.py
# эксп. 1 п.3 (после перезапуска с --enable-prefix-cache)
.venv/bin/python exp1_ttft.py --prefix

# эксп. 2 (снова без cache, -np 1)
.venv/bin/python exp2_tpot.py

# эксп. 3 (сервер с --np 4)
.venv/bin/python exp3_load.py
```

## Артефакты

`runs/<время>_<серия>/`:

- `config.yaml`, `metadata.json`, `results.jsonl`, `metrics.json`
- `traces/<id>.json` — сообщения, ответ, время, токены
- `figures/` — графики

MLflow: `mlflow.db`, эксперимент `LB1` (спаны LLM во вкладке Traces).
