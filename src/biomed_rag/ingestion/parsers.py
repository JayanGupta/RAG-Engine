"""XML/JSON parsers plus regex extraction of HR, p, N, dose, OS/PFS."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lxml import etree

from biomed_rag.models.documents import ExtractedStat, SourceDocument, TableBlob

NS = {
    "nlm": "https://jats.nlm.nih.gov/ns/archiving/1.0/",
}

HR_RE = re.compile(
    r"\b(?:HR|hazard ratio)\s*(?:=|:)?\s*([0-9]*\.?[0-9]+)\s*"
    r"(?:\((?:95%\s*)?CI[:\s]*([0-9.]+)\s*[-–to]+\s*([0-9.]+)\))?",
    re.I,
)
P_RE = re.compile(r"\bp\s*(?:value)?\s*(=|<|≤|<=|less than)\s*([0-9]*\.?[0-9]+)", re.I)
N_RE = re.compile(r"\b(?:n\s*=\s*|N\s*=\s*|sample size of\s*)(\d{2,5})\b")
DOSE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg(?:/kg)?(?:\s*(?:q\d+w|every\s+\d+\s+weeks)?)?)\b",
    re.I,
)
OS_RE = re.compile(
    r"\b(?:median\s+)?OS\s*(?:=|:)?\s*([0-9]*\.?[0-9]+)\s*(months?|mo)\b",
    re.I,
)
PFS_RE = re.compile(
    r"\b(?:median\s+)?PFS\s*(?:=|:)?\s*([0-9]*\.?[0-9]+)\s*(months?|mo)\b",
    re.I,
)

SECTION_ALIASES = {
    "abstract": "abstract",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusions": "discussion",
    "conclusion": "discussion",
}


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def extract_stats_from_text(text: str, table_id: str | None = None) -> list[ExtractedStat]:
    stats: list[ExtractedStat] = []
    for match in HR_RE.finditer(text):
        stats.append(
            ExtractedStat(metric="hr", value=match.group(1), table_id=table_id)
        )
    for match in P_RE.finditer(text):
        operator = (match.group(1) or "").strip()
        number = match.group(2)
        value = number if operator in {"", "="} else f"{operator}{number}" if operator in {"<", "≤", "<="} else number
        if operator == "less than":
            value = f"<{number}"
        stats.append(
            ExtractedStat(
                metric="p_value",
                value=value,
                table_id=table_id,
            )
        )
    for match in N_RE.finditer(text):
        stats.append(ExtractedStat(metric="n", value=match.group(1), table_id=table_id))
    for match in DOSE_RE.finditer(text):
        stats.append(
            ExtractedStat(
                metric="dose",
                value=match.group(1),
                unit=match.group(2),
                table_id=table_id,
            )
        )
    for match in OS_RE.finditer(text):
        stats.append(
            ExtractedStat(
                metric="median_os",
                value=match.group(1),
                unit=match.group(2),
                table_id=table_id,
            )
        )
    for match in PFS_RE.finditer(text):
        stats.append(
            ExtractedStat(
                metric="median_pfs",
                value=match.group(1),
                unit=match.group(2),
                table_id=table_id,
            )
        )
    return stats


def parse_tables(root: etree._Element, doc_id: str) -> tuple[list[TableBlob], list[ExtractedStat]]:
    tables: list[TableBlob] = []
    stats: list[ExtractedStat] = []
    for idx, table_wrap in enumerate(root.xpath(".//*[local-name()='table-wrap']"), start=1):
        table_id = table_wrap.get("id") or f"{doc_id}-t{idx}"
        caption_el = table_wrap.find(".//{*}caption")
        caption = _text(caption_el)
        table_el = table_wrap.find(".//{*}table")
        headers: list[str] = []
        rows: list[list[str]] = []
        html = ""
        if table_el is not None:
            html = etree.tostring(table_el, encoding="unicode")
            header_cells = table_el.xpath(".//*[local-name()='thead']//*[local-name()='th']")
            headers = [_text(c) for c in header_cells]
            for tr in table_el.xpath(".//*[local-name()='tbody']//*[local-name()='tr']"):
                cells = tr.xpath(".//*[local-name()='td'] | .//*[local-name()='th']")
                rows.append([_text(c) for c in cells])
        blob = TableBlob(
            table_id=table_id,
            doc_id=doc_id,
            caption=caption,
            headers=headers,
            rows=rows,
            html=html,
        )
        tables.append(blob)
        joined = caption + "\n" + " | ".join(headers)
        for row in rows:
            joined += "\n" + " | ".join(row)
        stats.extend(extract_stats_from_text(joined, table_id=table_id))
    return tables, stats


def parse_pubmed_xml(xml_bytes: bytes, therapeutic_area: str) -> list[SourceDocument]:
    root = etree.fromstring(xml_bytes)
    docs: list[SourceDocument] = []
    articles = root.xpath(".//*[local-name()='PubmedArticle']")
    if not articles:
        articles = [root]
    for article in articles:
        pmid = _text(article.find(".//{*}PMID"))
        title = _text(article.find(".//{*}ArticleTitle")) or "Untitled"
        abstract = _text(article.find(".//{*}Abstract"))
        journal = _text(article.find(".//{*}Journal/{*}Title")) or _text(
            article.find(".//{*}Title")
        )
        year_text = _text(article.find(".//{*}PubDate/{*}Year"))
        year = int(year_text) if year_text.isdigit() else None
        doi = ""
        for aid in article.xpath(".//*[local-name()='ArticleId']"):
            if (aid.get("IdType") or "").lower() == "doi":
                doi = _text(aid)
        mesh = [
            _text(m)
            for m in article.xpath(".//*[local-name()='DescriptorName']")
            if _text(m)
        ]
        nct_ids = re.findall(r"NCT\d{8}", abstract + " " + title, flags=re.I)
        stats = extract_stats_from_text(abstract)
        if not pmid:
            continue
        docs.append(
            SourceDocument(
                doc_id=pmid,
                source="pubmed",
                pmid=pmid,
                doi=doi or None,
                nct_id=nct_ids[0].upper() if nct_ids else None,
                title=title,
                year=year,
                journal=journal or None,
                section="abstract",
                text=abstract or title,
                extracted_stats=stats,
                trial_ids=[n.upper() for n in nct_ids],
                mesh_terms=mesh,
                therapeutic_area=therapeutic_area,
            )
        )
    return docs


def parse_pmc_xml(xml_bytes: bytes, therapeutic_area: str) -> tuple[list[SourceDocument], list[TableBlob]]:
    root = etree.fromstring(xml_bytes)
    article_meta = root.find(".//{*}article-meta")
    pmid = ""
    doi = ""
    pmcid = ""
    if article_meta is not None:
        for aid in article_meta.xpath(".//*[local-name()='article-id']"):
            pub_id_type = (aid.get("pub-id-type") or "").lower()
            if pub_id_type == "pmid":
                pmid = _text(aid)
            elif pub_id_type == "doi":
                doi = _text(aid)
            elif pub_id_type in {"pmc", "pmcid"}:
                pmcid = _text(aid)
    title = _text(root.find(".//{*}article-title")) or "Untitled"
    year_text = _text(root.find(".//{*}pub-date/{*}year"))
    year = int(year_text) if year_text.isdigit() else None
    journal = _text(root.find(".//{*}journal-title"))
    doc_id = pmid or pmcid or doi or "unknown"
    tables, table_stats = parse_tables(root, doc_id)
    nct_ids = re.findall(r"NCT\d{8}", _text(root), flags=re.I)

    sections: list[tuple[str, str]] = []
    abstract = _text(root.find(".//{*}abstract"))
    if abstract:
        sections.append(("abstract", abstract))
    for sec in root.xpath(".//*[local-name()='body']/*[local-name()='sec']"):
        heading = _text(sec.find("{*}title")).lower()
        mapped = "results"
        for key, value in SECTION_ALIASES.items():
            if key in heading:
                mapped = value
                break
        body = " ".join(
            t.strip()
            for t in sec.itertext()
            if t.strip() and _local(sec.tag) != "table"
        )
        if body:
            sections.append((mapped, body))
    if not sections:
        sections.append(("abstract", abstract or title))

    docs: list[SourceDocument] = []
    for section, text in sections:
        stats = extract_stats_from_text(text)
        if section == "results":
            stats = stats + table_stats
        docs.append(
            SourceDocument(
                doc_id=doc_id,
                source="pmc",
                pmid=pmid or None,
                doi=doi or None,
                nct_id=nct_ids[0].upper() if nct_ids else None,
                title=title,
                year=year,
                journal=journal or None,
                section=section,
                text=text,
                extracted_stats=_dedupe_stats(stats),
                trial_ids=sorted({n.upper() for n in nct_ids}),
                mesh_terms=[],
                therapeutic_area=therapeutic_area,
            )
        )
    return docs, tables


def parse_clinical_trial(record: dict, therapeutic_area: str) -> SourceDocument:
    protocol = record.get("protocolSection") or record
    ident = protocol.get("identificationModule", {})
    nct_id = ident.get("nctId") or record.get("nctId") or ""
    title = ident.get("officialTitle") or ident.get("briefTitle") or nct_id
    status = protocol.get("statusModule", {})
    year = None
    start = (status.get("startDateStruct") or {}).get("date") or ""
    if len(start) >= 4 and start[:4].isdigit():
        year = int(start[:4])
    desc = protocol.get("descriptionModule", {})
    design = protocol.get("designModule", {})
    arms_mod = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    ae_mod = protocol.get("adverseEventsModule", {})

    arms = []
    for arm in arms_mod.get("armGroups", []) or []:
        arms.append(
            f"{arm.get('label', '')}: {arm.get('type', '')} {arm.get('description', '')}"
        )
    interventions = []
    for item in arms_mod.get("interventions", []) or []:
        interventions.append(
            f"{item.get('name', '')} ({item.get('type', '')}) {item.get('description', '')}"
        )
    endpoints = []
    for item in outcomes.get("primaryOutcomes", []) or []:
        endpoints.append(item.get("measure", ""))
    aes: list[str] = []
    stats: list[ExtractedStat] = []
    for event in ae_mod.get("seriousEvents", []) or []:
        term = event.get("term") or event.get("organSystem") or "AE"
        aes.append(term)
        stats.append(ExtractedStat(metric="ae_grade", value=term, unit="serious"))

    phase = ",".join(design.get("phases", []) or []) or design.get("phase", "")
    enrollment = (design.get("enrollmentInfo") or {}).get("count")
    if enrollment:
        stats.append(ExtractedStat(metric="n", value=str(enrollment)))

    text_parts = [
        f"Clinical trial {nct_id}.",
        f"Title: {title}.",
        f"Phase: {phase}.",
        f"Brief summary: {desc.get('briefSummary', '')}",
        f"Arms: {'; '.join(arms)}",
        f"Interventions: {'; '.join(interventions)}",
        f"Primary endpoints: {'; '.join(endpoints)}",
        f"Serious adverse events: {'; '.join(aes[:20])}",
    ]
    text = "\n".join(p for p in text_parts if p)
    stats.extend(extract_stats_from_text(text))
    pmid = None
    refs = protocol.get("referencesModule", {})
    for ref in refs.get("references", []) or []:
        if ref.get("pmid"):
            pmid = str(ref["pmid"])
            break
    return SourceDocument(
        doc_id=nct_id,
        source="clinicaltrials",
        pmid=pmid,
        doi=None,
        nct_id=nct_id,
        title=title,
        year=year,
        journal="ClinicalTrials.gov",
        section="trial_record",
        text=text,
        extracted_stats=_dedupe_stats(stats),
        trial_ids=[nct_id] if nct_id else [],
        mesh_terms=[],
        therapeutic_area=therapeutic_area,
        extra={"phase": phase, "enrollment": enrollment},
    )


def _dedupe_stats(stats: list[ExtractedStat]) -> list[ExtractedStat]:
    seen: set[tuple] = set()
    out: list[ExtractedStat] = []
    for stat in stats:
        key = (stat.metric, stat.value, stat.arm, stat.unit, stat.table_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(stat)
    return out


def write_jsonl(path: Path, documents: list[SourceDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(doc.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[SourceDocument]:
    if not path.exists():
        return []
    docs: list[SourceDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            docs.append(SourceDocument.model_validate(json.loads(line)))
    return docs


def write_tables(tables_dir: Path, tables: list[TableBlob]) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[TableBlob]] = {}
    for table in tables:
        grouped.setdefault(table.doc_id, []).append(table)
    for doc_id, blobs in grouped.items():
        path = tables_dir / f"{doc_id}.json"
        path.write_text(
            json.dumps([b.model_dump() for b in blobs], indent=2),
            encoding="utf-8",
        )
