# BioMed-RAG Engine — Design Spec

Date: 2026-09-18
Status: Draft pending user review
Product: Clinical Intelligence and Medical Literature RAG application

## 1. Purpose

BioMed-RAG Engine is a production-grade retrieval-augmented generation system for oncology and immunotherapy literature. It ingests PubMed/PMC papers and ClinicalTrials.gov records, retrieves with hybrid biomedical search, synthesizes answers with mandatory PMID/DOI citations, and refuses diagnostic or treatment advice.

This is a literature synthesis tool, not a clinical decision support device. Every generated answer must include a clinical disclaimer.

## 2. Goals and Non-Goals

### Goals (v1)

- Ingest oncology/immunotherapy papers from PubMed Entrez and PMC Open Access XML, plus ClinicalTrials.gov API v2 JSON.
- Preserve structured trial statistics (p-values, hazard ratios, sample sizes, dosages) as JSON metadata linked to text chunks.
- Generate 50+ synthetic clinical QA pairs for evaluation, including adversarial cases.
- Semantic chunking plus Parent-Document retrieval.
- Hybrid search: dense biomedical embeddings + BM25, then cross-encoder rerank.
- LCEL generation pipeline with mandatory citations and disclaimers.
- Input/output guardrails that block non-medical and diagnostic queries.
- RAGAS benchmarking (Faithfulness, Answer Relevancy, Context Recall, Medical Groundedness).
- Streamlit dashboard with source citations, tables, chunk inspection, eval metrics.
- Optional LangSmith tracing via user-supplied keys.

### Non-Goals (v1)

- Live patient data, EHR, or PHI.
- Real-time diagnosis, dosing, or treatment recommendations.
- PDF OCR as the primary ingestion path.
- Multi-user authentication, RBAC, or tenancy.
- FastAPI split (monolith first; packages remain extractable).
- Using platform LLM keys (`MCAI_LLM_*`). User supplies `USER_LLM_*`.
- Installing or calling third-party agent runtimes.

## 3. Constraints

- Single exposed preview port. Streamlit is the only process users hit.
- No secrets in repo, logs, or UI. Placeholders in `.env.example` only.
- NCBI Entrez: identify with email; 3 requests/sec without key, 10/sec with key.
- All LLM/Cohere/LangSmith credentials are user-project variables, never Agent environment keys.
- Local-first vector store (embedded Qdrant). No required cloud vector DB.
- Python 3.11+, LangChain, Streamlit.

## 4. Recommended Approach

Streamlit in-process monolith.

Rationale: the preview environment exposes one port; Streamlit hosts UI and RAG in one process; LangChain LCEL does not need an HTTP API for v1. Package layout (`src/biomed_rag/`) keeps a later FastAPI extraction cheap.

Fallbacks:

- Vector store: Qdrant embedded (primary), Chroma (if Qdrant local client fails).
- Embeddings: `BAAI/bge-large-en-v1.5` (primary). Optional MedCPT query/article encoders if explicitly enabled in settings.
- Rerank: local `BAAI/bge-reranker-base`. Optional Cohere Rerank v3 if `USER_COHERE_API_KEY` is set.
- LLM: OpenAI-compatible chat via `USER_LLM_BASE_URL` + `USER_LLM_API_KEY` + `USER_LLM_MODEL`.

## 5. Architecture

```
User (Streamlit)
  -> Guardrail (input)
  -> EnsembleRetriever (dense + BM25)
  -> CrossEncoder / Cohere rerank
  -> Parent document + table metadata assembly
  -> LCEL prompt + LLM
  -> Citation parser + output policy
  -> UI (answer, PMID links, tables, chunk drawer)

Offline:
  PubMed Entrez / PMC OA XML / CT.gov v2
    -> parsers -> documents.jsonl
    -> SemanticChunker + ParentSplitter
    -> Qdrant collection + BM25 index

  Synthetic QA generator -> qa.jsonl -> RAGAS bench -> eval report
```

Process model: one Streamlit process. Ingestion and evaluation run from CLI scripts or a Streamlit admin page, not a second server.

## 6. Repository Layout

All files live at the git repository root (`/workspace`), not in a nested extra folder.

```
app.py
start.sh
pyproject.toml
requirements.txt
.env.example
README.md
config/
  settings.py
  prompts.py
src/biomed_rag/
    __init__.py
    models/
      documents.py
      qa.py
      retrieval.py
  ingestion/
    pubmed.py
    pmc.py
    clinical_trials.py
    parsers.py
    pipeline.py
  store/
    factory.py
    qdrant_store.py
    chroma_store.py
    bm25_index.py
  retrieval/
    chunking.py
    hybrid.py
    rerank.py
  generation/
    chain.py
    citations.py
  guardrails/
    input_filter.py
    output_policy.py
  eval/
    synthetic_qa.py
    ragas_bench.py
    groundedness.py
  observability/
    tracing.py
data/
  raw/
  processed/
  synthetic/
  eval/
tests/
  test_parsers.py
  test_guardrails.py
  test_citations.py
  test_chunking.py
  test_retrieval.py
  fixtures/
    sample_pmc.xml
    sample_pubmed.xml
    sample_trial.json
    sample_documents.jsonl
```

Application import root is `src/`. Streamlit `app.py` lives at repo root and imports `biomed_rag`.

## 7. Data Models

### 7.1 SourceDocument

```python
class ExtractedStat(BaseModel):
    metric: str          # hr | or | p_value | n | dose | ae_grade | median_os | median_pfs
    value: str           # keep original string to avoid false precision
    arm: str | None
    comparator: str | None
    unit: str | None
    table_id: str | None

class SourceDocument(BaseModel):
    doc_id: str          # pmid or nct_id
    source: str          # pubmed | pmc | clinicaltrials
    pmid: str | None
    doi: str | None
    nct_id: str | None
    title: str
    year: int | None
    journal: str | None
    section: str         # abstract | methods | results | discussion | trial_record
    text: str
    extracted_stats: list[ExtractedStat]
    trial_ids: list[str]
    mesh_terms: list[str]
    therapeutic_area: str  # oncology_immunotherapy
```

Persisted as JSONL at `data/processed/documents.jsonl`.

### 7.2 Chunk metadata (attached to vector points)

Required keys: `doc_id`, `pmid`, `doi`, `nct_id`, `title`, `year`, `section`, `parent_id`, `chunk_type` (`child` | `parent`), `extracted_stats`, `trial_ids`.

### 7.3 ClinicalQA (synthetic eval)

```python
class ClinicalQA(BaseModel):
    id: str
    question: str
    answer: str
    evidence_pmids: list[str]
    qtype: str  # comparison | adverse_event | statistic | mechanism | acronym | adversarial_oos | adversarial_dx
    adversarial: bool
    expected_behavior: str  # answer | refuse_oos | refuse_dx
```

Persisted at `data/synthetic/qa.jsonl`. Minimum 50 rows. Mix target:

- 15 multi-study / drug comparison
- 10 adverse-effect lookup
- 10 statistic / dose / HR lookup
- 5 mechanism / biomarker
- 5 acronym / synonym
- 5 out-of-scope adversarial
- 5 diagnostic-advice adversarial

## 8. Module Designs

### 8.1 Ingestion

Therapeutic area query (fixed for v1):

```
(immunotherapy OR "immune checkpoint" OR "PD-1" OR "PD-L1" OR pembrolizumab OR nivolumab OR atezolizumab)
AND (cancer OR oncology OR NSCLC OR melanoma OR carcinoma)
AND (clinical trial[pt] OR randomized)
```

Date window: last 10 years. Default fetch cap: 80 PubMed records + matching PMC XML when OA, plus 40 ClinicalTrials.gov studies for PD-1/PD-L1 in oncology. Caps are settings, not hardcoded magic in fetchers.

`pubmed.py`

- NCBI E-utilities: `esearch` then `efetch` XML.
- Headers/identity: `NCBI_EMAIL` required; `NCBI_API_KEY` optional.
- Rate limiter + exponential backoff on HTTP 429.
- Resume via `data/raw/pubmed/seen_pmids.json`.

`pmc.py`

- Map PMID to PMCID; download OA XML only.
- Prefer XML tables over PDF. PDF is out of v1 scope.

`clinical_trials.py`

- ClinicalTrials.gov API v2 (`studies` endpoint).
- Extract nct_id, phase, conditions, interventions, arms, primary endpoints, serious adverse events.

`parsers.py`

- Split PMC XML into sections.
- Convert XHTML/XML tables into `ExtractedStat` lists.
- Regex fallback on Results text for `HR`, `p =`, `n =`, `mg/kg` when table XML is missing.
- Never drop the raw table HTML; store under `data/processed/tables/{doc_id}.json`.

`pipeline.py`

- CLI: `python -m biomed_rag.ingestion.pipeline --limit 80`
- Writes jsonl, then optionally triggers indexing.

Failure policy: skip a record, log reason, continue. Pipeline succeeds if at least one document is written.

Offline demo corpus: `tests/fixtures/sample_documents.jsonl` (plus XML/JSON samples) ships with the repo so indexing, UI, and unit tests run without NCBI or ClinicalTrials.gov. Live fetch requires `NCBI_EMAIL`. Missing email is an error only for the live pipeline, not for Streamlit or tests.

### 8.2 Chunking and Store

`chunking.py`

- SemanticChunker on section text (percentile breakpoint, embedding-based).
- Parent/child split is implemented explicitly (not the LangChain ParentDocumentRetriever class wrapping the whole chain):
  - child: ~400 tokens, stored in the vector DB and BM25, used for search
  - parent: full section or ~2000 tokens, stored in a parent docstore keyed by `parent_id`, used for generation
- After hybrid search + rerank on children, lookup parents by `parent_id` and pass parents into the LLM.
- Tables are not semantically split. Each table becomes one parent-adjacent JSON blob referenced by `table_id`.

`store/factory.py`

- If `VECTOR_BACKEND=qdrant` (default): local Qdrant at `QDRANT_PATH` (default `data/qdrant`).
- If qdrant client import or init fails: Chroma at `data/chroma`.
- Collection name: `biomed_rag_oncology`.
- Embedding function from settings.

`bm25_index.py`

- In-memory BM25 over child chunk text plus a keyword field concatenating drug names, genes, acronyms.
- Serialized to `data/processed/bm25.pkl` so Streamlit restart does not require full reindex.

### 8.3 Hybrid retrieval and rerank

`hybrid.py`

- Dense: LangChain vector retriever, k=20.
- Sparse: BM25 retriever, k=20.
- `EnsembleRetriever` weights: dense 0.6, BM25 0.4.
- Deduplicate by `parent_id` before rerank.

`rerank.py`

- Local CrossEncoder `BAAI/bge-reranker-base`, keep top 5 parents.
- If `USER_COHERE_API_KEY` present, use Cohere rerank instead.
- Return parents + attached `extracted_stats` for the prompt.

### 8.4 Generation (LCEL)

`generation/chain.py` pipeline:

```
input_filter | retrieve_and_rerank | (empty retrieve -> insufficient-evidence, skip LLM) | build_prompt | llm | parse_citations | output_policy
```

System prompt rules (in `config/prompts.py`):

- Role: medical literature synthesizer, not a clinician.
- Use only provided context. If context is insufficient, say so.
- Every factual claim must be followed by a PMID and DOI when available, formatted `[PMID:xxxxxxxx]`.
- Never give personal diagnosis, prescriptions, or "you should take".
- Always append the disclaimer:

  > This output is a literature synthesis for research and educational use only. It is not medical advice, a diagnosis, or a treatment recommendation. Consult a qualified clinician.

`citations.py`

- Parse `[PMID:...]` and DOI patterns from the model output.
- Validate PMIDs against retrieved metadata.
- If zero valid citations and the query was not a refusal path, replace answer with insufficient-evidence template.

LLM client: LangChain `ChatOpenAI` pointed at `USER_LLM_BASE_URL`. Temperature 0.1. Max tokens 1200.

### 8.5 Guardrails

`input_filter.py` (RunnableLambda)

Allow if the query is about published biomedical literature, trials, drugs, genes, biomarkers, adverse events, or mechanisms.

Refuse with a fixed message (do not call LLM) when:

- Non-medical: programming, sports, politics, general chit-chat, math homework.
- Diagnostic intent: "do I have", "what should I take", "interpret my labs", "dose me".
- Requests to ignore the disclaimer or act as a doctor.

Implementation: keyword/regex + lightweight classifier prompt only if regex is inconclusive. Prefer regex-first to save tokens and reduce jailbreak surface. Classifier, when used, still cannot override an explicit diagnostic regex hit.

`output_policy.py`

- Strip imperative treatment language if the model emits it.
- Ensure disclaimer is present exactly once at the end.
- Block answers that invent PMIDs not in retrieved set.

### 8.6 Synthetic QA and RAGAS

`synthetic_qa.py`

- Uses LangChain structured output against `ClinicalQA`.
- Grounds generation on `documents.jsonl` so evidence PMIDs exist in the corpus.
- Adversarial items are written by templates, not the LLM, to keep them reliably off-policy.
- CLI: `python -m biomed_rag.eval.synthetic_qa --n 50`

`ragas_bench.py`

- Metrics: faithfulness, answer_relevancy, context_recall.
- Custom Medical Groundedness:
  - citation PMIDs subset of retrieved PMIDs
  - numeric claims (HR, p, n) fuzzy-match `extracted_stats` when present
  - refusals on adversarial items score 1.0 if expected_behavior is met
- Writes `data/eval/ragas_report.json`.
- Adversarial rows are excluded from RAGAS generation metrics and scored only on policy.

### 8.7 Streamlit UI

Single app, pages via sidebar:

1. Ask — query box, answer, disclaimer, latency.
2. Sources — title, year, PMID link `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`, DOI, snippet.
3. Extracted tables — dataframe of `extracted_stats` for cited docs.
4. Chunk inspector — expander per child chunk and its parent text.
5. Evaluation — load `ragas_report.json` if present.
6. Ingest — trigger/status for local pipeline (optional; primary path remains CLI).

No patient upload. No "treat me" affordance.

Telemetry: if `USER_LANGSMITH_API_KEY` and `USER_LANGSMITH_PROJECT` are set, enable LangChain tracing. Otherwise record latency and retrieved doc_ids in `st.session_state` only.

### 8.8 Observability

`observability/tracing.py`

- Configure LangSmith through user env vars only.
- Trace names: `biomed_rag.retrieve`, `biomed_rag.generate`.
- Never log API keys or full `.env`.

## 9. Configuration

`.env.example` (placeholders, no real values):

```
USER_LLM_API_KEY=your-api-key-here
USER_LLM_BASE_URL=https://api.example.com/v1
USER_LLM_MODEL=your-model-name
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=
USER_COHERE_API_KEY=
USER_LANGSMITH_API_KEY=
USER_LANGSMITH_PROJECT=biomed-rag-engine
VECTOR_BACKEND=qdrant
QDRANT_PATH=data/qdrant
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-base
USE_MEDCPT=false
```

`config/settings.py` uses pydantic-settings. Missing `USER_LLM_API_KEY` at query time shows a Streamlit error asking the user to fill `.env`; ingestion can still run without an LLM.

## 10. Error Handling

| Failure | Behavior |
|---|---|
| NCBI 429 / timeout | backoff, then skip batch |
| PMC not OA | keep PubMed abstract only |
| Empty retrieval | short-circuit before LLM; insufficient-evidence template |
| LLM down / bad key | Streamlit error, no retry storm |
| Qdrant init fail | fall back to Chroma once, log warning |
| Guardrail hit | fixed refusal, no retrieval |
| Missing citation | insufficient-evidence template |
| Eval with empty corpus | fail fast with message |

## 11. Testing

- Parsers: sample PMC XML fixture -> expected stats and sections.
- Guardrails: medical allow, oos refuse, diagnostic refuse, jailbreak-ish "ignore disclaimer".
- Citations: valid PMID kept; invented PMID stripped and answer replaced.
- Chunking: parent/child id linkage on a short fixture.
- Retrieval smoke: index 3 fixtures, query a drug name, expect BM25 hit.
- No live NCBI or paid LLM in default unit tests. Network tests marked optional.

## 12. Run and Preview

`start.sh` launches Streamlit on a single port (default 8501). After implementation, local preview uses the platform deploy-website skill. No reverse proxy is required because there is no separate frontend/backend.

Ingestion and eval are:

```
python -m biomed_rag.ingestion.pipeline --limit 80
python -m biomed_rag.eval.synthetic_qa --n 50
python -m biomed_rag.eval.ragas_bench
```

## 13. Implementation Phases

1. Scaffold: package layout, settings, models, requirements, `.env.example`.
2. Ingestion: PubMed, PMC, CT.gov, parsers, jsonl writer, fixtures.
3. Store and chunking: Qdrant/Chroma factory, semantic + parent split, BM25.
4. Hybrid retrieve + rerank.
5. LCEL chain, prompts, citations.
6. Guardrails input/output.
7. Synthetic QA generator.
8. RAGAS + Medical Groundedness.
9. Streamlit UI.
10. Tests, `start.sh`, README.

## 14. Success Criteria

- At least 50 synthetic QA pairs on disk.
- Hybrid retriever returns cited PMIDs for in-corpus statistic questions on the smoke fixture.
- Diagnostic and out-of-scope queries never reach the LLM (unit-tested).
- Every successful answer shows at least one PMID link and the disclaimer.
- RAGAS script produces a JSON report without crashing on the synthetic set.
- Streamlit serves Ask, Sources, Tables, Chunk inspector on one port.

## 15. Decisions Locked

- Therapeutic area: oncology / immunotherapy.
- UI: Streamlit monolith.
- Vector DB: Qdrant embedded, Chroma fallback.
- Embeddings: BGE-Large; MedCPT optional flag.
- Rerank: local cross-encoder; Cohere optional.
- Secrets: `USER_*` / `NCBI_*` only.
- PDF OCR: out of scope.
- FastAPI: out of scope for v1.
