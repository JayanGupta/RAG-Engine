from pathlib import Path

from biomed_rag.ingestion.parsers import (
    extract_stats_from_text,
    parse_clinical_trial,
    parse_pmc_xml,
    parse_pubmed_xml,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_hr_p_n_dose():
    text = "Median PFS 10.3 months (HR 0.50; 95% CI 0.37-0.68; p<0.001) n = 305. Pembrolizumab 200 mg."
    stats = extract_stats_from_text(text)
    metrics = {s.metric: s.value for s in stats}
    assert metrics["hr"] == "0.50"
    assert metrics["p_value"] == "<0.001"
    assert metrics["n"] == "305"
    assert metrics["dose"] == "200"


def test_parse_pubmed_xml():
    xml = (FIXTURES / "sample_pubmed.xml").read_bytes()
    docs = parse_pubmed_xml(xml, "oncology_immunotherapy")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.pmid == "27718847"
    assert doc.doi == "10.1056/NEJMoa1606774"
    assert "NCT02142738" in doc.trial_ids
    assert any(s.metric == "hr" for s in doc.extracted_stats)


def test_parse_pmc_xml_sections_and_tables():
    xml = (FIXTURES / "sample_pmc.xml").read_bytes()
    docs, tables = parse_pmc_xml(xml, "oncology_immunotherapy")
    sections = {d.section for d in docs}
    assert "abstract" in sections
    assert "methods" in sections
    assert "results" in sections
    assert tables
    assert tables[0].table_id == "t1"
    assert any(s.metric == "hr" for s in docs[0].extracted_stats) or any(
        s.metric == "hr" for d in docs for s in d.extracted_stats
    )


def test_parse_clinical_trial():
    import json

    record = json.loads((FIXTURES / "sample_trial.json").read_text(encoding="utf-8"))
    doc = parse_clinical_trial(record, "oncology_immunotherapy")
    assert doc.nct_id == "NCT02142738"
    assert doc.pmid == "27718847"
    assert doc.section == "trial_record"
    assert any(s.metric == "ae_grade" for s in doc.extracted_stats)
    assert any(s.metric == "n" and s.value == "305" for s in doc.extracted_stats)
