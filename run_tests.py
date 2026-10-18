"""Simple, zero-dependency test runner for RAG Engine."""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.test_documents import (
    test_document_creation_and_dict,
    test_save_and_load_jsonl,
    test_load_text_file,
)
from tests.test_chunking import (
    test_split_text_into_chunks,
    test_chunk_document,
    test_chunk_documents,
)
from tests.test_embeddings_and_vector_store import (
    test_hash_embeddings,
    test_cosine_similarity,
    test_vector_store_search,
)
from tests.test_bm25_and_hybrid import (
    test_bm25_search,
    test_reciprocal_rank_fusion,
    test_hybrid_retriever,
)
from tests.test_pipeline import (
    test_pipeline_demo_corpus,
    test_pipeline_query,
)


def run_all_tests():
    test_functions = [
        test_document_creation_and_dict,
        test_split_text_into_chunks,
        test_chunk_document,
        test_chunk_documents,
        test_hash_embeddings,
        test_cosine_similarity,
        test_vector_store_search,
        test_bm25_search,
        test_reciprocal_rank_fusion,
        test_hybrid_retriever,
        test_pipeline_demo_corpus,
        test_pipeline_query,
    ]

    import tempfile
    passed = 0
    failed = 0

    print("========================================")
    print("Running RAG Engine Test Suite")
    print("========================================")

    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  [PASS] {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    # Run tests that need tmp_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for test_fn in [test_save_and_load_jsonl, test_load_text_file]:
            try:
                test_fn(tmp_path)
                print(f"  [PASS] {test_fn.__name__}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {test_fn.__name__}: {e}")
                failed += 1

    print("========================================")
    print(f"Results: {passed} passed, {failed} failed")
    print("========================================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
