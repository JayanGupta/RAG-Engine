"""Heuristic faithfulness / relevancy / recall / groundedness over synthetic QA."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from config.settings import Settings, get_settings
from biomed_rag.eval.groundedness import medical_groundedness
from biomed_rag.eval.synthetic_qa import generate_synthetic_qa, load_qa
from biomed_rag.generation.chain import run_rag
from biomed_rag.ingestion.parsers import read_jsonl
from biomed_rag.models.qa import ClinicalQA
from biomed_rag.models.retrieval import RAGResult
from biomed_rag.store.factory import build_vector_store
from biomed_rag.store.indexer import index_documents


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in text.split() if len(t) > 2}


def heuristic_faithfulness(result: RAGResult) -> float:
    if result.refused:
        return 1.0
    ctx = " ".join((h.parent_text or h.text) for h in result.hits)
    if not ctx:
        return 0.0
    answer_tokens = _token_set(result.answer)
    ctx_tokens = _token_set(ctx)
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & ctx_tokens) / max(1, len(answer_tokens))
    cite = 1.0 if result.citations else 0.0
    return round(min(1.0, 0.7 * overlap + 0.3 * cite), 4)


def heuristic_answer_relevancy(result: RAGResult, question: str) -> float:
    if result.refused:
        return 1.0
    q = _token_set(question)
    a = _token_set(result.answer)
    if not q or not a:
        return 0.0
    return round(len(q & a) / len(q), 4)


def heuristic_context_recall(result: RAGResult, item: ClinicalQA) -> float:
    if item.adversarial:
        return 1.0 if result.refused else 0.0
    expected = set(item.evidence_pmids)
    if not expected:
        return 1.0 if result.hits else 0.0
    retrieved = {str(h.metadata.get("pmid")) for h in result.hits if h.metadata.get("pmid")}
    return round(len(expected & retrieved) / len(expected), 4)


def run_benchmark(
    settings: Settings | None = None,
    limit: int | None = None,
) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    docs = read_jsonl(settings.documents_path)
    if not docs:
        from biomed_rag.ingestion.pipeline import run_ingestion

        docs = run_ingestion(settings, live=False, index=True)
    bundle = build_vector_store(settings)
    if len(bundle.bm25.chunks) == 0:
        index_documents(docs, settings)
        bundle = build_vector_store(settings)
    qa_items = load_qa(settings.qa_path)
    if len(qa_items) < 50:
        qa_items = generate_synthetic_qa(settings, n=50, documents=docs)
    if limit:
        qa_items = qa_items[:limit]

    rows = []
    for item in qa_items:
        result = run_rag(item.question, settings=settings, bundle=bundle)
        row = {
            "id": item.id,
            "qtype": item.qtype,
            "adversarial": item.adversarial,
            "expected_behavior": item.expected_behavior,
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "citations": result.citations,
            "latency_ms": result.latency_ms,
            "faithfulness": heuristic_faithfulness(result),
            "answer_relevancy": heuristic_answer_relevancy(result, item.question),
            "context_recall": heuristic_context_recall(result, item),
            "medical_groundedness": medical_groundedness(result, item),
        }
        rows.append(row)

    gen_rows = [r for r in rows if not r["adversarial"]]
    adv_rows = [r for r in rows if r["adversarial"]]
    report = {
        "n": len(rows),
        "n_generation": len(gen_rows),
        "n_adversarial": len(adv_rows),
        "metrics": {
            "faithfulness": round(mean([r["faithfulness"] for r in gen_rows] or [0]), 4),
            "answer_relevancy": round(mean([r["answer_relevancy"] for r in gen_rows] or [0]), 4),
            "context_recall": round(mean([r["context_recall"] for r in gen_rows] or [0]), 4),
            "medical_groundedness": round(mean([r["medical_groundedness"] for r in rows] or [0]), 4),
            "adversarial_policy_accuracy": round(
                mean([r["medical_groundedness"] for r in adv_rows] or [0]), 4
            ),
        },
        "rows": rows,
        "ragas_backend": "heuristic_fallback",
    }
    try:
        report["ragas_backend"] = _try_ragas(qa_items, settings, bundle) or report["ragas_backend"]
    except Exception:
        pass
    settings.ragas_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _try_ragas(items: list[ClinicalQA], settings: Settings, bundle) -> str | None:
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
        from datasets import Dataset
    except Exception:
        return None
    if not settings.llm_configured():
        return None
    gen = [i for i in items if not i.adversarial][:8]
    questions, answers, contexts, gts = [], [], [], []
    for item in gen:
        result = run_rag(item.question, settings=settings, bundle=bundle)
        questions.append(item.question)
        answers.append(result.answer)
        contexts.append([h.parent_text or h.text for h in result.hits])
        gts.append(item.answer)
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": gts,
        }
    )
    evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    return "ragas"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = run_benchmark(limit=args.limit)
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {report['n']} rows")


if __name__ == "__main__":
    main()
