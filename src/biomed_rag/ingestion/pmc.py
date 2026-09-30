"""PMC Open Access XML client (full text and tables, not PDF OCR)."""

from __future__ import annotations

import time

import httpx
from lxml import etree

from config.settings import Settings
from biomed_rag.ingestion.parsers import parse_pmc_xml
from biomed_rag.models.documents import SourceDocument, TableBlob

OA_SVC = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


class PMCClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=45.0, follow_redirects=True)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def pmid_to_pmcid(self, pmid: str) -> str | None:
        try:
            response = self._client.get(
                IDCONV,
                params={"ids": pmid, "format": "json", "tool": "biomed_rag_engine", "email": self.settings.ncbi_email},
            )
            response.raise_for_status()
            records = response.json().get("records", [])
            if not records:
                return None
            pmcid = records[0].get("pmcid")
            return pmcid
        except httpx.HTTPError:
            return None

    def fetch_oa_xml(self, pmcid: str) -> bytes | None:
        params = {
            "db": "pmc",
            "id": pmcid,
            "retmode": "xml",
            "email": self.settings.ncbi_email,
            "tool": "biomed_rag_engine",
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        time.sleep(self.settings.ncbi_rate_limit())
        try:
            response = self._client.get(EFETCH, params=params)
            if response.status_code >= 400:
                return None
            root = etree.fromstring(response.content)
            if root.find(".//{*}article") is None and root.find(".//{*}pmc-articleset") is None:
                return None
            return response.content
        except (httpx.HTTPError, etree.XMLSyntaxError):
            return None

    def ingest_for_pmids(
        self, pmids: list[str]
    ) -> tuple[list[SourceDocument], list[TableBlob]]:
        docs: list[SourceDocument] = []
        tables: list[TableBlob] = []
        raw_dir = self.settings.raw_dir / "pmc"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for pmid in pmids:
            pmcid = self.pmid_to_pmcid(pmid)
            if not pmcid:
                continue
            xml_bytes = self.fetch_oa_xml(pmcid)
            if not xml_bytes:
                continue
            (raw_dir / f"{pmcid}.xml").write_bytes(xml_bytes)
            try:
                parsed, parsed_tables = parse_pmc_xml(
                    xml_bytes, self.settings.therapeutic_area
                )
            except etree.XMLSyntaxError:
                continue
            docs.extend(parsed)
            tables.extend(parsed_tables)
        return docs, tables
