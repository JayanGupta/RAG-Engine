"""NCBI E-utilities client: esearch + efetch PubMed abstracts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from config.settings import Settings
from biomed_rag.ingestion.parsers import parse_pubmed_xml
from biomed_rag.models.documents import SourceDocument

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _params(self, extra: dict) -> dict:
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "tool": "biomed_rag_engine",
            "email": self.settings.ncbi_email,
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        params.update(extra)
        return params

    def _get(self, path: str, params: dict) -> httpx.Response:
        delay = self.settings.ncbi_rate_limit()
        last_error: Exception | None = None
        for attempt in range(4):
            time.sleep(delay * (attempt + 1))
            try:
                response = self._client.get(f"{EUTILS}/{path}", params=params)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"NCBI request failed: {last_error}")

    def search_ids(self, query: str, limit: int) -> list[str]:
        response = self._get(
            "esearch.fcgi",
            self._params(
                {
                    "term": query,
                    "retmax": str(limit),
                    "mindate": self.settings.pubmed_mindate,
                    "datetype": "pdat",
                    "usehistory": "n",
                }
            ),
        )
        from lxml import etree

        root = etree.fromstring(response.content)
        return [el.text for el in root.xpath(".//Id") if el.text]

    def fetch_xml(self, pmids: list[str]) -> bytes:
        return self._get(
            "efetch.fcgi",
            self._params({"id": ",".join(pmids), "rettype": "abstract"}),
        ).content

    def ingest(
        self,
        limit: int | None = None,
        seen_path: Path | None = None,
    ) -> list[SourceDocument]:
        if not self.settings.ncbi_email:
            raise ValueError("NCBI_EMAIL is required for live PubMed ingestion")
        limit = limit or self.settings.pubmed_fetch_limit
        seen_path = seen_path or (self.settings.raw_dir / "pubmed" / "seen_pmids.json")
        seen: set[str] = set()
        if seen_path.exists():
            seen = set(json.loads(seen_path.read_text(encoding="utf-8")))
        ids = [i for i in self.search_ids(self.settings.pubmed_query, limit) if i not in seen]
        docs: list[SourceDocument] = []
        raw_dir = self.settings.raw_dir / "pubmed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(ids), 20):
            batch = ids[i : i + 20]
            try:
                xml_bytes = self.fetch_xml(batch)
                (raw_dir / f"batch_{batch[0]}.xml").write_bytes(xml_bytes)
                parsed = parse_pubmed_xml(xml_bytes, self.settings.therapeutic_area)
                docs.extend(parsed)
                seen.update(batch)
            except Exception:
                continue
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps(sorted(seen)), encoding="utf-8")
        return docs
