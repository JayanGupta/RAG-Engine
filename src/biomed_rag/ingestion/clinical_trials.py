"""ClinicalTrials.gov API v2 client for completed PD-1/PD-L1 oncology studies."""

from __future__ import annotations

import json

import httpx

from config.settings import Settings
from biomed_rag.ingestion.parsers import parse_clinical_trial
from biomed_rag.models.documents import SourceDocument

CTGOV = "https://clinicaltrials.gov/api/v2/studies"

DEFAULT_QUERY = (
    "AREA[ConditionSearch](cancer OR melanoma OR NSCLC) AND "
    "AREA[InterventionName](pembrolizumab OR nivolumab OR atezolizumab OR "
    '"PD-1" OR "PD-L1")'
)


class ClinicalTrialsClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=45.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, limit: int | None = None) -> list[dict]:
        limit = limit or self.settings.ctgov_fetch_limit
        params = {
            "query.term": DEFAULT_QUERY,
            "pageSize": min(limit, 50),
            "format": "json",
            "filter.overallStatus": "COMPLETED",
        }
        try:
            response = self._client.get(CTGOV, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError:
            return []
        return payload.get("studies", [])[:limit]

    def ingest(self, limit: int | None = None) -> list[SourceDocument]:
        studies = self.search(limit)
        raw_dir = self.settings.raw_dir / "clinicaltrials"
        raw_dir.mkdir(parents=True, exist_ok=True)
        docs: list[SourceDocument] = []
        for study in studies:
            try:
                doc = parse_clinical_trial(study, self.settings.therapeutic_area)
            except Exception:
                continue
            nct = doc.nct_id or doc.doc_id
            (raw_dir / f"{nct}.json").write_text(
                json.dumps(study)[:500000], encoding="utf-8"
            )
            docs.append(doc)
        return docs
