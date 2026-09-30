"""Fetch and parse papers into SourceDocument records.

pipeline.run_ingestion(live=False) loads the demo corpus.
pipeline.run_ingestion(live=True) hits PubMed, PMC, and ClinicalTrials.gov.
"""

from biomed_rag.ingestion.pipeline import run_ingestion

__all__ = ["run_ingestion"]
