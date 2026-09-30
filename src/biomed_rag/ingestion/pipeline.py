"""Build documents.jsonl from the demo corpus or live NCBI / CT.gov.

Default (no --live): copy tests/fixtures/sample_documents.jsonl.
With --live: PubMed abstracts, PMC OA XML, ClinicalTrials.gov records.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config.settings import Settings, get_settings
from biomed_rag.ingestion.clinical_trials import ClinicalTrialsClient
from biomed_rag.ingestion.parsers import read_jsonl, write_jsonl, write_tables
from biomed_rag.ingestion.pmc import PMCClient
from biomed_rag.ingestion.pubmed import PubMedClient
from biomed_rag.models.documents import SourceDocument, TableBlob

FIXTURE_DOCUMENTS = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_documents.jsonl"
)


def load_demo_corpus(settings: Settings) -> list[SourceDocument]:
    if not FIXTURE_DOCUMENTS.exists():
        return []
    docs = read_jsonl(FIXTURE_DOCUMENTS)
    write_jsonl(settings.documents_path, docs)
    return docs


def run_ingestion(
    settings: Settings | None = None,
    limit: int | None = None,
    live: bool = False,
    index: bool = False,
) -> list[SourceDocument]:
    settings = settings or get_settings()
    settings.ensure_dirs()
    docs: list[SourceDocument] = []
    tables: list[TableBlob] = []

    if not live:
        docs = load_demo_corpus(settings)
        if index and docs:
            from biomed_rag.store.indexer import index_documents

            index_documents(docs, settings)
        return docs

    pubmed = PubMedClient(settings)
    pmc = PMCClient(settings)
    ctgov = ClinicalTrialsClient(settings)
    try:
        pubmed_docs = pubmed.ingest(limit=limit)
        docs.extend(pubmed_docs)
        pmids = [d.pmid for d in pubmed_docs if d.pmid]
        pmc_docs, pmc_tables = pmc.ingest_for_pmids(pmids)
        docs.extend(pmc_docs)
        tables.extend(pmc_tables)
        docs.extend(ctgov.ingest(limit=settings.ctgov_fetch_limit))
    finally:
        pubmed.close()
        pmc.close()
        ctgov.close()

    if not docs:
        docs = load_demo_corpus(settings)
    else:
        write_jsonl(settings.documents_path, docs)
        write_tables(settings.tables_dir, tables)

    if index and docs:
        from biomed_rag.store.indexer import index_documents

        index_documents(docs, settings)
    return docs


def seed_demo_tables(settings: Settings) -> None:
    fixture_tables = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tables"
    if not fixture_tables.exists():
        return
    settings.tables_dir.mkdir(parents=True, exist_ok=True)
    for src in fixture_tables.glob("*.json"):
        shutil.copy(src, settings.tables_dir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest biomedical literature")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true", help="Fetch from NCBI/CT.gov")
    parser.add_argument("--index", action="store_true", help="Index after ingest")
    args = parser.parse_args()
    settings = get_settings()
    docs = run_ingestion(settings, limit=args.limit, live=args.live, index=args.index)
    seed_demo_tables(settings)
    print(f"ingested {len(docs)} documents -> {settings.documents_path}")


if __name__ == "__main__":
    main()
