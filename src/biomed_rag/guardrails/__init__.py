"""Safety gates around retrieval and generation.

filter_query() runs before search. apply_output_policy() runs after citations.
"""

from biomed_rag.guardrails.input_filter import filter_query
from biomed_rag.guardrails.output_policy import apply_output_policy

__all__ = ["filter_query", "apply_output_policy"]
