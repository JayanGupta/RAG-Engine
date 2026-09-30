"""Synthetic QA set and heuristic RAGAS-style benchmark."""

from biomed_rag.eval.synthetic_qa import generate_synthetic_qa
from biomed_rag.eval.ragas_bench import run_benchmark

__all__ = ["generate_synthetic_qa", "run_benchmark"]
