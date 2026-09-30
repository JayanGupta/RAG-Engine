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

python3 -m biomed_rag.ingestion.pipeline
python3 - <<'PY'
from config.settings import get_settings
from biomed_rag.ingestion.parsers import read_jsonl
from biomed_rag.store.indexer import index_documents
from biomed_rag.eval.synthetic_qa import generate_synthetic_qa, load_qa

settings = get_settings()
docs = read_jsonl(settings.documents_path)
if docs:
    index_documents(docs, settings)
if len(load_qa(settings.qa_path)) < 50:
    generate_synthetic_qa(settings, n=50, documents=docs)
print("bootstrap complete")
PY

exec python3 -m streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0 --browser.gatherUsageStats false
