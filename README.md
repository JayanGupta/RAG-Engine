<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=34&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=900&height=95&lines=Modular+RAG+Engine+%E2%80%A2+Built+from+Scratch;Hybrid+Search+%E2%80%A2+Dense+Embeddings+%2B+BM25;Reciprocal+Rank+Fusion+%2B+RAG+Evaluation" alt="Typing SVG" /></a>

<br/>

<!-- Animated Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=220&section=header&text=RAG%20Engine&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Production-Grade%20Hybrid%20Retrieval-Augmented%20Generation%20System&descSize=18&descAlignY=55&descColor=58a6ff" width="100%"/>

<br/>

<!-- Badges Row 1: Repository Status -->
![GitHub last commit](https://img.shields.io/github/last-commit/JayanGupta/RAG-Engine?style=for-the-badge&color=00d9ff&labelColor=0d1117)
![GitHub repo size](https://img.shields.io/github/repo-size/JayanGupta/RAG-Engine?style=for-the-badge&color=7c3aed&labelColor=0d1117)
![GitHub stars](https://img.shields.io/github/stars/JayanGupta/RAG-Engine?style=for-the-badge&color=fbbf24&labelColor=0d1117)
![GitHub forks](https://img.shields.io/github/forks/JayanGupta/RAG-Engine?style=for-the-badge&color=34d399&labelColor=0d1117)

<br/>

<!-- Badges Row 2: Core Data Technologies -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![BM25](https://img.shields.io/badge/Search-Okapi_BM25-00D9FF?style=for-the-badge&logo=speedtest&logoColor=white)
![RRF](https://img.shields.io/badge/Ranking-RRF_Fusion-7c3aed?style=for-the-badge&logo=apache&logoColor=white)

<br/>

<!-- Badges Row 3: Developer & Quality Tooling -->
![Pytest](https://img.shields.io/badge/Pytest-Passed-22c55e?style=for-the-badge&logo=pytest&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

<br/><br/>

<!-- Profile Views Counter -->
![Profile Views](https://komarev.com/ghpvc/?username=JayanGupta&label=Repository+Views&color=00d9ff&style=for-the-badge)

</div>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## About This Repository

> *A clean, modular, and educational **Retrieval-Augmented Generation (RAG)** engine built from the ground up — without restrictive third-party framework lock-ins, black-box abstractions, or bloated dependencies.*

Large Language Models (LLMs) excel at reasoning but suffer from **temporal knowledge cutoffs** and **hallucination**. This repository demonstrates how a production-ready RAG system solves these limitations by turning open-ended generation into an **open-book synthesis problem**:

1. **Retrieve**: Extract the most contextually relevant passages using **Hybrid Search** (combining Dense Semantic Vectors with Sparse Lexical BM25).
2. **Fuse**: Rank and synthesize multi-modal retrieval candidates using **Reciprocal Rank Fusion (RRF)**.
3. **Augment**: Ground LLM synthesis strictly on retrieved evidence with formatted source citations.
4. **Evaluate**: Score the output automatically for **Faithfulness** (factual grounding) and **Answer Relevancy**.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion & Chunking"]
        A["Raw Knowledge Corpus<br/>(Markdown, JSONL, Text)"] --> B["Document Loader & Normalizer"]
        B --> C["Sliding-Window Chunking<br/>(Overlapping Boundaries)"]
    end

    subgraph Indexing["2. Dual Hybrid Indexing"]
        C --> D["Dense Vector Index<br/>(Cosine Semantic Embeddings)"]
        C --> E["Sparse Lexical Index<br/>(Okapi BM25 Term Matching)"]
    end

    subgraph Retrieval["3. Hybrid Retrieval & Rank Fusion"]
        Q["User Query"] --> F["Vector Similarity Search"]
        Q --> G["BM25 Keyword Search"]
        D -.-> F
        E -.-> G
        F --> H["Reciprocal Rank Fusion (RRF)<br/>Score = Σ w / (k + rank)"]
        G --> H
        H --> I["Top-K Ranked Passages"]
    end

    subgraph Generation["4. Synthesis & Evaluation"]
        I --> J["Context Prompt Injector"]
        Q --> J
        J --> K["OpenAI LLM / Offline Extractive Fallback"]
        K --> L["Cited Grounded Response"]
        L --> M["Automated Evaluation<br/>(Faithfulness & Relevancy)"]
    end

    style Ingestion fill:#0d1117,stroke:#00d9ff,stroke-width:2px,color:#ffffff
    style Indexing fill:#0d1117,stroke:#34d399,stroke-width:2px,color:#ffffff
    style Retrieval fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style Generation fill:#0d1117,stroke:#fbbf24,stroke-width:2px,color:#ffffff
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Core Technical Concepts

### 1. Sliding Window Chunking
Embedding entire documents into single vectors dilutes granular details. We decompose documents into focused passages (e.g., 300 words) with overlapping spans (50 words) to guarantee conceptual continuity across chunk boundaries.

### 2. Dual Hybrid Search
* **Dense Semantic Embeddings**: Captures conceptual affinity (e.g., `"climate variation"` matches `"global warming"`). Includes zero-download offline hash embeddings and optional SentenceTransformers.
* **Sparse Okapi BM25**: Excels at verbatim keywords, technical acronyms, serial numbers, and exact entities (e.g., `NIRCam`, `15 millikelvin`, `ISO-8601`) where pure vector models often struggle.

### 3. Reciprocal Rank Fusion (RRF)
To unify rankings without requiring brittle calibration of raw vector distances and BM25 scores, we combine rankings using Reciprocal Rank Fusion:

$$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

### 4. Zero-Cost Offline Extractive Fallback
If an `OPENAI_API_KEY` is provided, the engine utilizes GPT models to synthesize fluent answers. When running locally without API credentials, the pipeline gracefully defaults to an **offline extractive synthesis engine**, making the entire codebase 100% free and runnable offline.

### 5. Automated RAG Triad Evaluation
* **Faithfulness**: Measures whether all assertions made in the generated answer are strictly substantiated by the retrieved context.
* **Answer Relevancy**: Verifies that the synthesized answer directly addresses the user query.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Quick Start & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JayanGupta/RAG-Engine.git
cd RAG-Engine
```

### 2. Create Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Optional Environment Configuration
Copy the template configuration:
```bash
cp .env.example .env
```
*(Optional: Insert your `OPENAI_API_KEY` in `.env` for generative answers. If left empty, the engine uses local extractive synthesis).*

### 4. Launch the Interactive Streamlit Web UI
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the visual dashboard, adjust retrieval weights, and inspect chunk rankings.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Programmatic Python Usage

```python
from src.rag_engine.pipeline import RAGPipeline
from src.rag_engine.documents import load_documents_from_jsonl

# 1. Initialize Pipeline & Ingest Knowledge Base
pipeline = RAGPipeline()
docs = load_documents_from_jsonl("data/documents.jsonl")
pipeline.index_documents(docs)

# 2. Query the Hybrid Search Engine
query = "What temperature must superconducting qubits be maintained at?"
result = pipeline.query(query, top_k=3)

# 3. Inspect Response, Citations & Metrics
print("Generated Answer:\n", result.answer)
print("\nSources Cited:")
for source in result.sources:
    print(f"- {source.title} (RRF Score: {source.score:.4f})")

print(f"\nFaithfulness: {result.metrics['faithfulness']:.2f}")
print(f"Relevancy:    {result.metrics['relevancy']:.2f}")
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Running the Verification Suite

Run unit and integration tests across chunking, dual indexing, BM25, and hybrid fusion:

```bash
# Run via pytest
pytest tests/ -v

# Or use the standalone runner
python run_tests.py
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Repository File Structure

```
RAG-Engine/
|-- app.py                                 # Streamlit visual interactive application
|-- pyproject.toml                         # Project metadata and tool configurations
|-- requirements.txt                       # Python dependencies
|-- run_tests.py                           # Standalone test runner script
|-- start.sh                               # Container entrypoint script
|-- README.md                              # Master repository documentation
|-- .env.example                           # Template environment configuration
|
|-- config/                                # System configuration & prompts
|   |-- settings.py                        # Search weights, chunk sizes & defaults
|   `-- prompts.py                         # LLM system & context augmentation prompts
|
|-- data/                                  # Demo corpus and benchmark knowledge bases
|   `-- documents.jsonl                    # Multi-domain articles (Science, Tech, Quantum)
|
|-- experiment/                            # Research & educational scratchpads
|   `-- rag_pipeline.ipynb                 # Interactive Jupyter step-by-step notebook
|
|-- src/rag_engine/                        # Core Modular RAG Engine
|   |-- chunking.py                        # Overlapping sliding window text splitter
|   |-- documents.py                       # Document abstractions & serializers
|   |-- embeddings.py                      # Dense vector embedding adapters
|   |-- vector_store.py                    # In-memory cosine similarity store
|   |-- bm25_search.py                     # Okapi BM25 sparse keyword search engine
|   |-- hybrid_search.py                   # Reciprocal Rank Fusion (RRF) algorithm
|   |-- generator.py                       # LLM and local extractive synthesis
|   |-- eval.py                            # Faithfulness & relevancy metrics
|   `-- pipeline.py                        # Unified end-to-end orchestration pipeline
|
`-- tests/                                 # Complete Pytest test suite
    |-- test_chunking.py                   # Chunk split & overlap unit tests
    |-- test_documents.py                  # Document validation tests
    |-- test_embeddings_and_vector_store.py# Cosine similarity verification
    |-- test_bm25_and_hybrid.py            # BM25 & RRF fusion scoring tests
    `-- test_pipeline.py                   # End-to-end integration tests
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Contributing

Contributions, bug fixes, and suggestions are welcome:

1. Fork the Project (`https://github.com/JayanGupta/RAG-Engine/fork`)
2. Create your Feature Branch (`git checkout -b feature/AdvancedReranker`)
3. Commit your Changes (`git commit -m "feat: add cross-encoder reranker support"`)
4. Push to the Branch (`git push origin feature/AdvancedReranker`)
5. Open a Pull Request

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Show Your Support

Give a star if this repository helped your understanding of Retrieval-Augmented Generation!

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=JayanGupta/RAG-Engine&type=Date&theme=dark)](https://star-history.com/#JayanGupta/RAG-Engine&Date)

</div>

---

<div align="center">

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=120&section=footer" width="100%"/>

<br/>

<b>Engineered for AI Innovation by <a href="https://github.com/JayanGupta">Jayan Gupta</a></b>

<br/><br/>

<img src="https://forthebadge.com/images/badges/built-with-love.svg" />
<img src="https://forthebadge.com/images/badges/made-with-python.svg" />
<img src="https://forthebadge.com/images/badges/open-source.svg" />

</div>
