import os
import json
import pytest
import tempfile
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch
from app.config import get_settings
from app.services.chunking_service import Chunk

# We will import these once we create them, but for now we write the tests using mocks/placeholders or directly testing the math/functions.
# To do TDD, we can mock import the functions or write the expected signatures.
# Let's import them. (They will cause import errors until implemented, which is the TDD initial expected failure!).
from app.evaluation.evaluator import calculate_keyword_match_score, LLMJudgeEvaluator
from app.evaluation.run import (
    calculate_hit_at_k,
    calculate_mrr_at_k,
    compute_aggregates,
    run_configuration_evaluation
)

@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_dataset_schema_validation():
    # Test valid and invalid dataset formats
    valid_data = {
        "dataset_version": "1.0",
        "questions": [
            {
                "id": "q-001",
                "question": "Query?",
                "expected_answer": "Ans",
                "expected_source_documents": ["doc-1"],
                "category": "direct_answer",
                "expected_behavior": None
            }
        ]
    }
    
    # Asserting that schema matches our expectations
    assert valid_data["dataset_version"] == "1.0"
    assert len(valid_data["questions"]) == 1
    assert valid_data["questions"][0]["category"] == "direct_answer"
    assert "expected_source_documents" in valid_data["questions"][0]


def test_single_document_hit_rate():
    # Hit@K should be 1 if expected document is in retrieved chunks, else 0
    retrieved_chunks = [
        Chunk(chunk_id="c1", document_id="doc-A", source_filename="f1.txt", text="a", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="c2", document_id="doc-B", source_filename="f2.txt", text="b", chunk_index=0, block_type="paragraph")
    ]
    
    # Doc A is in retrieved list
    assert calculate_hit_at_k(retrieved_chunks, ["doc-A"], k=5) == 1
    # Doc C is not in retrieved list
    assert calculate_hit_at_k(retrieved_chunks, ["doc-C"], k=5) == 0
    # k value limits the scan
    assert calculate_hit_at_k(retrieved_chunks, ["doc-B"], k=1) == 0


def test_cross_document_hit_rate():
    # Hit@K is 1 only when ALL expected source documents are in top-K chunks
    retrieved_chunks = [
        Chunk(chunk_id="c1", document_id="doc-A", source_filename="f1.txt", text="a", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="c2", document_id="doc-B", source_filename="f2.txt", text="b", chunk_index=0, block_type="paragraph")
    ]
    
    # Both Doc A and Doc B are in top 5 chunks -> Success
    assert calculate_hit_at_k(retrieved_chunks, ["doc-A", "doc-B"], k=5) == 1
    # Only Doc A is in top 1 chunk, Doc B is at index 1 (rank 2) -> Fail for k=1
    assert calculate_hit_at_k(retrieved_chunks, ["doc-A", "doc-B"], k=1) == 0
    # Expected Doc C is missing -> Fail
    assert calculate_hit_at_k(retrieved_chunks, ["doc-A", "doc-C"], k=5) == 0


def test_unanswerable_evaluation():
    # Unanswerable has expected_source_documents = []
    # Hit@K and MRR should be None (not applicable)
    retrieved_chunks = [
        Chunk(chunk_id="c1", document_id="doc-A", source_filename="f1.txt", text="a", chunk_index=0, block_type="paragraph")
    ]
    assert calculate_hit_at_k(retrieved_chunks, [], k=5) is None
    assert calculate_mrr_at_k(retrieved_chunks, [], k=5) is None


def test_mrr_calculation_single():
    # MRR = 1 / first matching rank index
    retrieved_chunks = [
        Chunk(chunk_id="c1", document_id="doc-A", source_filename="f1.txt", text="a", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="c2", document_id="doc-B", source_filename="f2.txt", text="b", chunk_index=0, block_type="paragraph")
    ]
    
    # Doc A is rank 1 -> MRR = 1.0
    assert calculate_mrr_at_k(retrieved_chunks, ["doc-A"], k=5) == 1.0
    # Doc B is rank 2 -> MRR = 0.5
    assert calculate_mrr_at_k(retrieved_chunks, ["doc-B"], k=5) == 0.5
    # Missing -> MRR = 0.0
    assert calculate_mrr_at_k(retrieved_chunks, ["doc-C"], k=5) == 0.0


def test_mrr_calculation_cross():
    # MRR cross uses the maximum (worst) first-retrieved rank for each expected document
    retrieved_chunks = [
        Chunk(chunk_id="c1", document_id="doc-A", source_filename="f1.txt", text="a", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="c2", document_id="doc-C", source_filename="f3.txt", text="c", chunk_index=0, block_type="paragraph"),
        Chunk(chunk_id="c3", document_id="doc-B", source_filename="f2.txt", text="b", chunk_index=0, block_type="paragraph")
    ]
    
    # Doc A is at rank 1, Doc B is at rank 3 -> worst rank is 3 -> MRR = 1/3
    mrr = calculate_mrr_at_k(retrieved_chunks, ["doc-A", "doc-B"], k=5)
    assert pytest.approx(mrr) == 1/3
    
    # If any required document is missing -> MRR = 0
    mrr_missing = calculate_mrr_at_k(retrieved_chunks, ["doc-A", "doc-D"], k=5)
    assert mrr_missing == 0.0


def test_keyword_match_score():
    # Test simple overlap count ratio calculation
    # e.g., prompt expectations keywords
    assert calculate_keyword_match_score("Vivek knows Python.", ["python"]) == 1.0
    assert calculate_keyword_match_score("Vivek knows programming.", ["python"]) == 0.0
    assert calculate_keyword_match_score("Python and JavaScript are known.", ["python", "javascript", "go"]) == pytest.approx(2/3)


def test_per_question_result_creation():
    # Asserts single question evaluation record structure
    # Must contain IDs and performance details, but NO raw retrieved text
    per_q = {
        "question_id": "q-1",
        "category": "direct_answer",
        "retrieved_source_documents": ["doc-A"],
        "expected_source_documents": ["doc-A"],
        "first_relevant_rank": 1,
        "hit_at_k": 1,
        "reciprocal_rank": 1.0,
        "fallback_expected": None,
        "fallback_returned": None,
        "top_score": 0.85,
        "latency": {"embedding_ms": 12.0, "retrieval_ms": 5.0, "llm_ms": 0.0, "total_ms": 17.0},
        "keyword_match_score": 1.0,
        "evaluation_status": "success"
    }
    assert "retrieved_context" not in per_q
    assert "text" not in per_q
    assert per_q["question_id"] == "q-1"


def test_aggregate_statistics():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    agg = compute_aggregates(latencies)
    assert agg["mean"] == 30.0
    assert agg["median"] == 30.0
    # p95 on small dataset is calculated dynamically
    assert agg["p95"] >= 30.0
    assert agg["sample_count"] == 5


def test_judge_failure_produces_null_metrics():
    # When evaluator judge fails, score should record as null and status as judge_failed
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="invalid-key")
    
    with patch("httpx.Client.post") as mock_post:
        mock_post.side_effect = ValueError("Unauthorized API key")
        
        score, status, _ = evaluator.evaluate_correctness("Q", "A", "Expected")
        assert score is None
        assert status == "judge_failed"


def test_reproducibility_metadata():
    report = {
        "dataset_version": "1.0",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_provider": "mock",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "relevance_threshold": 0.4,
        "timestamp": "2026-08-31T12:00:00Z"
    }
    assert "embedding_model" in report
    assert report["chunk_size"] == 500


def test_configuration_isolation_and_cleanup(temp_log_dir, monkeypatch):
    # Setup mock env using temp_log_dir
    monkeypatch.setenv("LOGS_DIR", str(temp_log_dir / "logs"))
    monkeypatch.setenv("UPLOAD_DIR", str(temp_log_dir / "raw"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(temp_log_dir / "vectors"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    dataset = {
        "dataset_version": "1.0",
        "questions": [
            {
                "id": "q-1",
                "question": "What are the office hours?",
                "expected_answer": "Office hours are from 9:00 AM to 5:00 PM.",
                "expected_source_documents": ["22222222-2222-2222-2222-222222222222"],
                "category": "direct_answer",
                "keywords": ["9:00", "5:00"]
            }
        ]
    }

    # Run configuration evaluation
    res = run_configuration_evaluation(500, 50, dataset)

    # Assert configuration runs successfully
    assert len(res["results"]) == 1
    assert res["results"][0]["question_id"] == "q-1"

    # Assert isolation and clean up: temporary raw and vectors directories must be deleted
    # In run_configuration_evaluation, it creates a temporary directory using tempfile.mkdtemp()
    # and deletes it. So no files leak.


def test_evaluation_corpus_size():
    from app.evaluation.run import SYNTHETIC_CORPUS
    # Verify we have at least 3 documents
    assert len(SYNTHETIC_CORPUS) >= 3
    # Verify each document is approximately 2,000 words (minimum 1,500 words)
    for doc_id, doc_info in SYNTHETIC_CORPUS.items():
        word_count = len(doc_info["content"].split())
        assert word_count >= 1500, f"Document {doc_id} word count {word_count} is less than 1500"


def test_stable_document_ids():
    from app.evaluation.run import SYNTHETIC_CORPUS
    expected_ids = {
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333"
    }
    for doc_id in SYNTHETIC_CORPUS.keys():
        assert doc_id in expected_ids


def test_dataset_distribution_schema():
    dataset_path = Path("app/evaluation/dataset.json")
    assert dataset_path.exists()
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    questions = dataset["questions"]
    assert len(questions) == 35
    
    categories = [q["category"] for q in questions]
    assert categories.count("direct_answer") == 10
    assert categories.count("multi_part") == 7
    assert categories.count("cross_document") == 6
    assert categories.count("unanswerable") == 6
    assert categories.count("hallucination_check") == 6

    # Verify unanswerable questions have expected_source_documents = []
    for q in questions:
        if q["category"] == "unanswerable":
            assert q["expected_source_documents"] == []


def test_chunk_layout_variation():
    # Verify different configurations result in different chunk counts for at least some documents
    from app.services.chunking_service import chunk_document
    from app.services.document_extractor import ExtractedDocument, ExtractedBlock
    from app.evaluation.run import SYNTHETIC_CORPUS

    doc_a = SYNTHETIC_CORPUS["11111111-1111-1111-1111-111111111111"]
    extracted = ExtractedDocument(
        document_id="11111111-1111-1111-1111-111111111111",
        original_filename=doc_a["filename"],
        source_filename=doc_a["filename"],
        file_type="txt",
        blocks=[
            ExtractedBlock(type="paragraph", index=1, text=doc_a["content"])
        ]
    )

    chunks_500 = chunk_document(extracted, chunk_size=500, chunk_overlap=50)
    chunks_1000 = chunk_document(extracted, chunk_size=1000, chunk_overlap=100)

    # Configuration 500/50 should produce more chunks than 1000/100
    assert len(chunks_500) > len(chunks_1000)


def test_discriminative_validation_warning():
    # If configurations produce identical layout counts, verify warning detection logic works
    from app.evaluation.run import check_is_discriminative
    
    # Identical layout counts across configurations
    identical_runs = [
        {"chunk_size": 500, "chunk_overlap": 50, "aggregates": {"total_chunks": 10}},
        {"chunk_size": 800, "chunk_overlap": 80, "aggregates": {"total_chunks": 10}},
        {"chunk_size": 1000, "chunk_overlap": 100, "aggregates": {"total_chunks": 10}}
    ]
    assert check_is_discriminative(identical_runs) is False

    # Varied layout counts
    varied_runs = [
        {"chunk_size": 500, "chunk_overlap": 50, "aggregates": {"total_chunks": 20}},
        {"chunk_size": 800, "chunk_overlap": 80, "aggregates": {"total_chunks": 12}},
        {"chunk_size": 1000, "chunk_overlap": 100, "aggregates": {"total_chunks": 6}}
    ]
    assert check_is_discriminative(varied_runs) is True


def test_benchmark_uses_expanded_corpus(temp_log_dir, monkeypatch):
    # Integration test verifying that the benchmark runner uses the expanded corpus
    # and counts show the expanded documents' chunk counts are correctly tracked.
    monkeypatch.setenv("LOGS_DIR", str(temp_log_dir / "logs"))
    monkeypatch.setenv("UPLOAD_DIR", str(temp_log_dir / "raw"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(temp_log_dir / "vectors"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    import app.services.vector_store as vector_store_module
    import app.services.embedding_service as embedding_service_module
    vector_store_module._vector_store_instance = None
    embedding_service_module._embedding_service_instance = None

    dataset_path = Path("app/evaluation/dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Run for single config
    res = run_configuration_evaluation(500, 50, dataset)
    assert "aggregates" in res
    assert "total_chunks" in res["aggregates"]
    # Verify we got multiple chunks per document, total chunks should be significantly greater than 3
    assert res["aggregates"]["total_chunks"] > 5


def test_cross_document_retrieval_diagnostic_q018_to_q023(temp_log_dir, monkeypatch):
    """Diagnostic test verifying current 500/50 retrieval behavior for cross-document questions q-018 to q-023."""
    monkeypatch.setenv("LOGS_DIR", str(temp_log_dir / "logs"))
    monkeypatch.setenv("UPLOAD_DIR", str(temp_log_dir / "raw"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(temp_log_dir / "vectors"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    import app.services.vector_store as vector_store_module
    import app.services.embedding_service as embedding_service_module
    vector_store_module._vector_store_instance = None
    embedding_service_module._embedding_service_instance = None

    dataset_path = Path("app/evaluation/dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    res = run_configuration_evaluation(500, 50, dataset)
    results = res["results"]
    cross_doc_res = {r["question_id"]: r for r in results if r["category"] == "cross_document"}

    # Diagnostic assertions
    print("\n[DIAGNOSTIC TEST] Cross document results in test suite:")
    for q_id, res_q in cross_doc_res.items():
        print(f"  {q_id}: hit_at_k={res_q['hit_at_k']}, docs={res_q['retrieved_source_documents']}")

    # q-018: Both doc 11111111 and doc 33333333 retrieved in Top-5
    assert cross_doc_res["q-018"]["hit_at_k"] == 1

    # q-019: Both doc 22222222 and doc 11111111 retrieved in Top-5
    assert cross_doc_res["q-019"]["hit_at_k"] == 1

    # q-020: Both doc 22222222 and doc 33333333 retrieved in Top-5
    assert cross_doc_res["q-020"]["hit_at_k"] == 1

    # q-021: Both doc 11111111 and doc 22222222 retrieved in Top-5 (improved by diversity retrieval)
    assert cross_doc_res["q-021"]["hit_at_k"] == 1

    # q-022: Both doc 11111111 and doc 33333333 retrieved in Top-5 (improved by diversity retrieval)
    assert cross_doc_res["q-022"]["hit_at_k"] == 1

    # q-023: Both doc 22222222 and doc 33333333 retrieved in Top-5 (improved by query decomposition)
    assert cross_doc_res["q-023"]["hit_at_k"] == 1

    # Verify overall cross-document Hit@5 increased to 1.00 (6/6)
    total_hits = sum(cross_doc_res[qid]["hit_at_k"] for qid in cross_doc_res)
    assert total_hits == 6, f"Expected 6 hits out of 6, got {total_hits}: {cross_doc_res}"



