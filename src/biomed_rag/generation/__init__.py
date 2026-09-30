"""Turn retrieved parents into a cited answer.

run_rag() is the only entry the UI needs. citations.py strips invented PMIDs.
"""

from biomed_rag.generation.chain import run_rag, build_chain_inputs

__all__ = ["run_rag", "build_chain_inputs"]
