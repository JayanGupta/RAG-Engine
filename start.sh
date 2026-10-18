#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi

PORT="${PORT:-8501}"

echo "Starting RAG Engine..."
python3 - <<'PY'
from rag_engine.pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.load_demo_corpus()
print("RAG Engine initialized successfully.")
PY

exec python3 -m streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0 --browser.gatherUsageStats false
