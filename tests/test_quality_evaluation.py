import json
import os
import tempfile
import time
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.config import get_settings
from app.evaluation.evaluator import LLMJudgeEvaluator
from app.evaluation.metrics import (
    calculate_claim_faithfulness_score,
    calculate_citation_correctness_score,
    evaluate_refusal_correctness,
    aggregate_quality_metrics
)
from app.evaluation.run_quality import run_quality_evaluation

@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_quality_metrics_correctness():
    # Mock LLMJudgeEvaluator correctness evaluation
    evaluator = LLMJudgeEvaluator(provider="mock", model="gpt-4o-mini")
    
    # Matching answer
    score, status, reason = evaluator.evaluate_correctness(
        question="What is the annual leave allowance?",
        answer="Employees receive 25 annual leave days.",
        expected_answer="Employees receive 25 annual leave days."
    )
    assert status == "success"
    assert score is not None
    assert score == 1.0
    assert reason is None

    # Completely different answer
    score, status, reason = evaluator.evaluate_correctness(
        question="What is the annual leave allowance?",
        answer="Only corporate-issued computers are allowed.",
        expected_answer="Employees receive 25 annual leave days."
    )
    assert status == "success"
    assert score is not None
    assert score < 0.2


def test_quality_metrics_claim_faithfulness():
    from app.evaluation.metrics import ClaimEvaluation, CitationValidation
    
    # 2 claims, 1 supported, 1 unsupported
    claims = [
        ClaimEvaluation(
            claim="Employees get 25 annual leave days.",
            supported=True,
            supporting_source_ids=["11111111-1111-1111-1111-111111111111"],
            citations=[]
        ),
        ClaimEvaluation(
            claim="Employees get a free car.",
            supported=False,
            supporting_source_ids=[],
            citations=[]
        )
    ]
    score = calculate_claim_faithfulness_score(claims)
    assert score == 0.5


def test_quality_metrics_citation():
    from app.evaluation.metrics import ClaimEvaluation, CitationValidation

    # Test cases:
    # 1. Perfect citation (exists, retrieved, supports)
    # 2. Citation document does not exist
    # 3. Citation document exists but was not retrieved
    # 4. Citation document exists, retrieved, but does not support
    claims = [
        ClaimEvaluation(
            claim="Leave is 25 days.",
            supported=True,
            supporting_source_ids=["doc-1"],
            citations=[
                CitationValidation(
                    citation_text="[1]",
                    document_id="doc-1",
                    document_exists=True,
                    document_was_retrieved=True,
                    source_supports_claim=True
                )
            ]
        ),
        ClaimEvaluation(
            claim="IT rules are strict.",
            supported=True,
            supporting_source_ids=["doc-2"],
            citations=[
                CitationValidation(
                    citation_text="[2]",
                    document_id="doc-99",  # Does not exist
                    document_exists=False,
                    document_was_retrieved=False,
                    source_supports_claim=False
                )
            ]
        ),
        ClaimEvaluation(
            claim="IT rules are strict part 2.",
            supported=True,
            supporting_source_ids=["doc-2"],
            citations=[
                CitationValidation(
                    citation_text="[3]",
                    document_id="doc-3",  # Exists, but not retrieved
                    document_exists=True,
                    document_was_retrieved=False,
                    source_supports_claim=False
                ),
                CitationValidation(
                    citation_text="[4]",
                    document_id="doc-4",  # Exists, retrieved, does not support
                    document_exists=True,
                    document_was_retrieved=True,
                    source_supports_claim=False
                )
            ]
        )
    ]

    score = calculate_citation_correctness_score(claims)
    # Total criteria tested:
    # Citation 1: exists(T), retrieved(T), supports(T) -> 3/3 correct
    # Citation 2: exists(F), retrieved(F), supports(F) -> 0/3 correct
    # Citation 3: exists(T), retrieved(F), supports(F) -> 1/3 correct
    # Citation 4: exists(T), retrieved(T), supports(F) -> 2/3 correct
    # Total correct criteria = 3 + 0 + 1 + 2 = 6 out of 12. Score = 0.5.
    assert score == 0.5


def test_quality_metrics_refusal():
    # Correct refusal (expected and returned)
    correct, returned = evaluate_refusal_correctness(
        category="unanswerable",
        answer="I could not find information about this in the provided documents.",
        relevance_threshold=0.4,
        top_score=0.35
    )
    assert correct is True
    assert returned is True

    # Unnecessary refusal (valid question, but refused anyway)
    correct, returned = evaluate_refusal_correctness(
        category="direct_answer",
        answer="I could not find information about this in the provided documents.",
        relevance_threshold=0.4,
        top_score=0.75
    )
    assert correct is False
    assert returned is True

    # Incorrect non-refusal (unanswerable category, but returned some hallucination answer)
    correct, returned = evaluate_refusal_correctness(
        category="unanswerable",
        answer="Vivek Shinde is a master in Go.",
        relevance_threshold=0.4,
        top_score=0.35
    )
    assert correct is False
    assert returned is False


def test_judge_output_validation():
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    
    # 1. Malformed JSON
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not-json-content"}}]
        }
        mock_post.return_value = mock_resp
        
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == "malformed_json"

    # 2. Invalid score range (e.g. 9.5)
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 9.5}"}}]
        }
        mock_post.return_value = mock_resp
        
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == "invalid_score"

    # 3. Missing required field
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"wrong_key\": 1.0}"}}]
        }
        mock_post.return_value = mock_resp
        
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == "missing_required_field"


def test_mock_offline_runs():
    # Verifies that mock evaluation works fully offline
    evaluator = LLMJudgeEvaluator(provider="mock", model="gpt-4o-mini")
    
    # Verifies correct claims evaluation mock
    claims, status, reason = evaluator.evaluate_claims_and_citations(
        question="What is the leave?",
        answer="Leave is 25 days.",
        context="Leave is 25 days.",
        retrieved_ids=["11111111-1111-1111-1111-111111111111"]
    )
    assert status == "success"
    assert len(claims) > 0
    assert claims[0].supported is True


def test_reproducibility_metadata(temp_log_dir, monkeypatch):
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
                "expected_source_documents": ["11111111-1111-1111-1111-111111111111"],
                "category": "direct_answer",
                "keywords": ["9:00", "5:00"]
            }
        ]
    }

    # Run mock quality evaluation
    report = run_quality_evaluation("mock", "gpt-4o-mini", dataset)
    assert "metadata" in report
    metadata = report["metadata"]
    assert metadata["judge_provider"] == "mock"
    assert metadata["judge_model"] == "gpt-4o-mini"
    assert metadata["judge_temperature"] == 0.0
    assert metadata["dataset_version"] == "1.0"
    assert "evaluation_timestamp" in metadata


def test_all_evaluations_failing_produces_none_aggregates():
    # Construct results where all judge calls failed
    results = [
        {
            "category": "direct_answer",
            "judge_status": "judge_failed",
            "latency": {"total_ms": 10.0}
        },
        {
            "category": "multi_part",
            "judge_status": "judge_failed",
            "latency": {"total_ms": 15.0}
        }
    ]
    
    aggs = aggregate_quality_metrics(results)
    overall = aggs["overall"]
    assert overall["generation"]["mean_correctness"] is None
    assert overall["generation"]["mean_faithfulness"] is None
    assert overall["performance"]["sample_count"] == 2


def test_mixed_quality_evaluations():
    results = [
        {
            "category": "direct_answer",
            "judge_status": "success",
            "answer_correctness_score": 0.8,
            "faithfulness_score": 1.0,
            "context_relevance_score": 0.5,
            "citation_correctness_score": 0.9,
            "latency": {"total_ms": 10.0}
        },
        {
            "category": "direct_answer",
            "judge_status": "judge_failed",
            "latency": {"total_ms": 12.0}
        }
    ]
    
    aggs = aggregate_quality_metrics(results)
    overall = aggs["overall"]
    # correctness should be exactly 0.8 since the failed run is excluded
    assert overall["generation"]["mean_correctness"] == 0.8
    # correctness category breakdown should also exclude the failed run
    assert aggs["categories"]["direct_answer"]["generation"]["mean_correctness"] == 0.8


def test_missing_api_key_aborts_instantly():
    # Instantiate without key
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="")
    with patch("httpx.Client.post") as mock_post:
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == "missing_required_field"
        # Confirm no HTTP request was made
        mock_post.assert_not_called()


def test_reproducibility_api_key_loading(monkeypatch):
    # Verify key is loaded from EVAL_QUALITY_API_KEY
    monkeypatch.setenv("EVAL_QUALITY_API_KEY", "loaded-quality-key")
    get_settings.cache_clear()
    
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini")
    assert evaluator.api_key == "loaded-quality-key"


@pytest.mark.parametrize(
    "status_code, expected_reason",
    [
        (401, "http_401_unauthorized"),
        (403, "http_403_forbidden"),
        (429, "http_429_rate_limited"),
        (400, "http_400_bad_request"),
        (404, "http_404_not_found"),
        (500, "http_5xx_server_error"),
        (503, "http_5xx_server_error"),
        (418, "http_error_other"),
    ],
)
def test_openai_http_status_classification(status_code, expected_reason):
    import httpx
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    req = httpx.Request("POST", "http://test-judge-url")
    mock_resp = httpx.Response(status_code=status_code, request=req)

    with patch("httpx.Client.post", return_value=mock_resp):
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == expected_reason

        score_rel, status_rel, reason_rel = evaluator.evaluate_context_relevance("Q", "C")
        assert score_rel is None
        assert status_rel == "judge_failed"
        assert reason_rel == expected_reason

        claims, status_claims, reason_claims = evaluator.evaluate_claims_and_citations("Q", "A", "C", ["doc1"])
        assert claims == []
        assert status_claims == "judge_failed"
        assert reason_claims == expected_reason


def test_latency_measurement_accounting(temp_log_dir, monkeypatch):
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
                "expected_source_documents": ["11111111-1111-1111-1111-111111111111"],
                "category": "direct_answer",
                "keywords": ["9:00", "5:00"]
            }
        ]
    }

    report = run_quality_evaluation("mock", "gpt-4o-mini", dataset)
    res = report["results"][0]
    lat = res["latency"]
    assert "rag_pipeline_ms" in lat
    assert "pipeline_ms" in lat
    assert "judge_ms" in lat
    assert "total_ms" in lat
    assert lat["rag_pipeline_ms"] >= 0.0
    assert lat["judge_ms"] >= 0.0
    assert lat["total_ms"] >= lat["rag_pipeline_ms"]
    assert pytest.approx(lat["total_ms"]) == lat["rag_pipeline_ms"] + lat["judge_ms"]


def test_gemini_provider_configuration_and_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-123")
    get_settings.cache_clear()

    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash")
    assert evaluator.provider == "gemini"
    assert evaluator.model == "gemini-2.5-flash"
    assert evaluator.api_key == "test-gemini-key-123"
    assert evaluator.api_url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_gemini_client_url_model_key_passed_correctly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-abc")
    get_settings.cache_clear()

    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash")
    
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 4.0}"}}]
        }
        mock_post.return_value = mock_resp

        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert status == "success"
        assert score == 0.75

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        target_url = call_args[0][0]
        headers = call_args[1]["headers"]
        payload = call_args[1]["json"]

        assert target_url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert headers["Authorization"] == "Bearer test-gemini-key-abc"
        assert payload["model"] == "gemini-2.5-flash"


def test_gemini_successful_parsing():
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="valid-key")

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 5.0}"}}]
        }
        mock_post.return_value = mock_resp
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert status == "success"
        assert score == 1.0

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 3.0}"}}]
        }
        mock_post.return_value = mock_resp
        score, status, reason = evaluator.evaluate_context_relevance("Q", "C")
        assert status == "success"
        assert score == 0.5

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "claims": [{
                            "claim": "Claim 1",
                            "supported": True,
                            "supporting_source_ids": ["doc-1"],
                            "citations": []
                        }]
                    })
                }
            }]
        }
        mock_post.return_value = mock_resp
        claims, status, reason = evaluator.evaluate_claims_and_citations("Q", "A", "C", ["doc-1"])
        assert status == "success"
        assert len(claims) == 1
        assert claims[0].claim == "Claim 1"


@pytest.mark.parametrize(
    "status_code, expected_reason",
    [
        (400, "http_400_bad_request"),
        (401, "http_401_unauthorized"),
        (403, "http_403_forbidden"),
        (429, "http_429_rate_limited"),
        (500, "http_5xx_server_error"),
    ],
)
def test_gemini_http_error_handling(status_code, expected_reason):
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    mock_resp = httpx.Response(status_code=status_code, request=req)

    with patch("httpx.Client.post", return_value=mock_resp):
        score, status, reason = evaluator.evaluate_correctness("Q", "A", "E")
        assert score is None
        assert status == "judge_failed"
        assert reason == expected_reason


def test_unsupported_provider_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM Judge provider"):
        LLMJudgeEvaluator(provider="unsupported", model="some-model")


def test_api_keys_never_exposed_in_serialized_report(temp_log_dir, monkeypatch):
    monkeypatch.setenv("LOGS_DIR", str(temp_log_dir / "logs"))
    monkeypatch.setenv("UPLOAD_DIR", str(temp_log_dir / "raw"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(temp_log_dir / "vectors"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET_GEMINI_API_KEY_999")
    monkeypatch.setenv("EVAL_QUALITY_API_KEY", "SECRET_EVAL_QUALITY_API_KEY_888")
    get_settings.cache_clear()

    dataset = {
        "dataset_version": "1.0",
        "questions": [
            {
                "id": "q-1",
                "question": "What are the office hours?",
                "expected_answer": "Office hours are from 9:00 AM to 5:00 PM.",
                "expected_source_documents": ["11111111-1111-1111-1111-111111111111"],
                "category": "direct_answer",
                "keywords": ["9:00", "5:00"]
            }
        ]
    }

    report = run_quality_evaluation("mock", "gemini-2.5-flash", dataset)
    serialized_str = json.dumps(report)

    assert "SECRET_GEMINI_API_KEY_999" not in serialized_str
    assert "SECRET_EVAL_QUALITY_API_KEY_888" not in serialized_str
    assert "Authorization" not in serialized_str
    assert "Bearer" not in serialized_str


def test_gemini_requests_are_throttled():
    evaluator = LLMJudgeEvaluator(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="test-key",
        rpm_limit=30,
    )

    with patch("httpx.Client.post") as mock_post, patch("time.sleep") as mock_sleep:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 5.0}"}}]
        }
        mock_post.return_value = mock_resp

        LLMJudgeEvaluator._gemini_last_request_time = time.perf_counter()
        evaluator.evaluate_correctness("Q1", "A1", "E1")

        mock_sleep.assert_called_once()
        wait_time = mock_sleep.call_args[0][0]
        assert 0.0 < wait_time <= 2.0


def test_openai_requests_not_throttled():
    evaluator = LLMJudgeEvaluator(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        rpm_limit=15,
    )

    with patch("httpx.Client.post") as mock_post, patch("time.sleep") as mock_sleep:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"score\": 5.0}"}}]
        }
        mock_post.return_value = mock_resp

        evaluator.evaluate_correctness("Q1", "A1", "E1")
        evaluator.evaluate_correctness("Q2", "A2", "E2")

        mock_sleep.assert_not_called()


def test_mock_requests_not_throttled():
    evaluator = LLMJudgeEvaluator(provider="mock", model="gpt-4o-mini", rpm_limit=15)

    with patch("time.sleep") as mock_sleep:
        evaluator.evaluate_correctness("Q1", "A1", "E1")
        evaluator.evaluate_correctness("Q2", "A2", "E2")

        mock_sleep.assert_not_called()


def test_rpm_configuration_respected(monkeypatch):
    monkeypatch.setenv("EVAL_QUALITY_RPM_LIMIT", "60")
    get_settings.cache_clear()

    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="test-key")
    assert evaluator.rpm_limit == 60


def test_evaluate_all_mock_success():
    evaluator = LLMJudgeEvaluator(provider="mock", model="gpt-4o-mini")
    corr, rel, claims, status, reason = evaluator.evaluate_all(
        question="What are office hours?",
        answer="Office hours are 9am to 5pm.",
        expected_answer="Office hours are 9am to 5pm.",
        context="Office hours are 9am to 5pm.",
        retrieved_ids=["11111111-1111-1111-1111-111111111111"]
    )
    assert status == "success"
    assert corr == 1.0
    assert rel == 0.5
    assert isinstance(claims, list)
    assert reason is None


def test_evaluate_all_openai_success():
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_score": 5.0,
                        "context_relevance_score": 4.0,
                        "claims": [{
                            "claim": "Office hours are 9 to 5",
                            "supported": True,
                            "supporting_source_ids": ["doc-1"],
                            "citations": [{
                                "citation_text": "[1]",
                                "document_id": "doc-1",
                                "document_exists": True,
                                "document_was_retrieved": True,
                                "source_supports_claim": True
                            }]
                        }]
                    })
                }
            }]
        }
        mock_post.return_value = mock_resp

        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])

        assert mock_post.call_count == 1
        assert status == "success"
        assert corr == 1.0
        assert rel == 0.75
        assert len(claims) == 1
        assert claims[0].claim == "Office hours are 9 to 5"
        assert claims[0].supported is True


def test_evaluate_all_gemini_success():
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_score": 3.0,
                        "context_relevance_score": 5.0,
                        "claims": []
                    })
                }
            }]
        }
        mock_post.return_value = mock_resp

        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])

        assert mock_post.call_count == 1
        assert status == "success"
        assert corr == 0.5
        assert rel == 1.0
        assert claims == []

        call_args = mock_post.call_args
        assert call_args[0][0] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert call_args[1]["headers"]["Authorization"] == "Bearer gemini-key"


def test_evaluate_all_malformed_json():
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not-valid-json"}}]
        }
        mock_post.return_value = mock_resp

        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert corr is None
        assert rel is None
        assert claims == []
        assert status == "judge_failed"
        assert reason == "malformed_json"


def test_evaluate_all_missing_required_fields():
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"correctness_score": 5.0})}}]
        }
        mock_post.return_value = mock_resp

        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert corr is None
        assert rel is None
        assert claims == []
        assert status == "judge_failed"
        assert reason == "missing_required_field"


def test_evaluate_all_invalid_score():
    evaluator = LLMJudgeEvaluator(provider="openai", model="gpt-4o-mini", api_key="test-key")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_score": 99.0,
                        "context_relevance_score": 4.0,
                        "claims": []
                    })
                }
            }]
        }
        mock_post.return_value = mock_resp

        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert corr is None
        assert rel is None
        assert claims == []
        assert status == "judge_failed"
        assert reason == "invalid_score"


@pytest.mark.parametrize(
    "status_code, expected_reason",
    [
        (400, "http_400_bad_request"),
        (401, "http_401_unauthorized"),
        (403, "http_403_forbidden"),
        (429, "http_429_rate_limited"),
        (500, "http_5xx_server_error"),
        (503, "http_5xx_server_error"),
    ],
)
def test_evaluate_all_http_errors(status_code, expected_reason):
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    mock_resp = httpx.Response(status_code=status_code, request=req)

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert corr is None
        assert rel is None
        assert claims == []
        assert status == "judge_failed"
        assert reason == expected_reason


def test_evaluate_all_timeout_and_connection_error():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout")):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "timeout"

    with patch("httpx.Client.post", side_effect=Exception("Connection failed")):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "connection_error"


def test_evaluate_all_gemini_throttled_single_call():
    LLMJudgeEvaluator._gemini_last_request_time = 0.0
    evaluator = LLMJudgeEvaluator(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="test-key",
        rpm_limit=30,
    )

    with patch("httpx.Client.post") as mock_post, patch("time.sleep") as mock_sleep:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_score": 5.0,
                        "context_relevance_score": 5.0,
                        "claims": []
                    })
                }
            }]
        }
        mock_post.return_value = mock_resp

        evaluator.evaluate_all("Q1", "A1", "E1", "C1", ["doc1"])
        evaluator.evaluate_all("Q2", "A2", "E2", "C2", ["doc2"])

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()


def test_legacy_evaluator_methods_remain_functional():
    evaluator = LLMJudgeEvaluator(provider="mock", model="gpt-4o-mini")
    
    corr_score, corr_stat, corr_reason = evaluator.evaluate_correctness("Q", "A", "A")
    assert corr_stat == "success"
    assert corr_score == 1.0

    rel_score, rel_stat, rel_reason = evaluator.evaluate_context_relevance("Q", "Q")
    assert rel_stat == "success"
    assert rel_score == 1.0

    claims, claims_stat, claims_reason = evaluator.evaluate_claims_and_citations("Q", "A", "C", ["doc1"])
    assert claims_stat == "success"
    assert isinstance(claims, list)


def test_run_quality_uses_evaluate_all_and_not_individual_methods(temp_log_dir, monkeypatch):
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
                "expected_source_documents": ["11111111-1111-1111-1111-111111111111"],
                "category": "direct_answer",
                "keywords": ["9:00", "5:00"]
            }
        ]
    }

    with patch("app.evaluation.run_quality.evaluate_refusal_correctness", return_value=(True, False)), \
         patch.object(LLMJudgeEvaluator, "evaluate_all", return_value=(1.0, 1.0, [], "success", None)) as mock_eval_all, \
         patch.object(LLMJudgeEvaluator, "evaluate_correctness") as mock_corr, \
         patch.object(LLMJudgeEvaluator, "evaluate_context_relevance") as mock_rel, \
         patch.object(LLMJudgeEvaluator, "evaluate_claims_and_citations") as mock_claims:

        report = run_quality_evaluation("mock", "gpt-4o-mini", dataset)

        assert mock_eval_all.call_count == 1
        assert mock_corr.call_count == 0
        assert mock_rel.call_count == 0
        assert mock_claims.call_count == 0


def test_gemini_diagnostic_classification_429_rate_limit():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    body = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded. Please try again in 4s.", "type": "rate_limit"}})
    mock_resp = httpx.Response(status_code=429, request=req, text=body)

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "http_429_rate_limited"
        diag = evaluator.last_diagnostic
        assert diag is not None
        assert diag["status_code"] == 429
        assert diag["classified_error_type"] == "http_429_rate_limited"
        assert diag["provider"] == "gemini"
        assert diag["model"] == "gemini-2.5-flash"
        assert diag["endpoint_url"] == "generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert diag["api_key_present"] is True
        assert diag["timeout_seconds"] == 10.0
        assert isinstance(diag["request_latency_ms"], float)
        assert diag["error_details"]["type"] == "rate_limit"
        assert diag["error_details"]["code"] == 429


def test_gemini_diagnostic_classification_429_quota_exhausted():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    body = json.dumps({"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}})
    mock_resp = httpx.Response(status_code=429, request=req, text=body)

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "http_429_quota_exhausted"
        diag = evaluator.last_diagnostic
        assert diag is not None
        assert diag["status_code"] == 429
        assert diag["classified_error_type"] == "http_429_quota_exhausted"
        assert diag["error_details"]["type"] == "RESOURCE_EXHAUSTED"


def test_gemini_diagnostic_classification_5xx_server_error():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    body = json.dumps({"error": {"code": 503, "message": "The model is overloaded.", "type": "unavailable"}})
    mock_resp = httpx.Response(status_code=503, request=req, text=body)

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "http_5xx_server_error"
        diag = evaluator.last_diagnostic
        assert diag["status_code"] == 503
        assert diag["classified_error_type"] == "http_5xx_server_error"


def test_gemini_diagnostic_classification_timeout():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "timeout"
        diag = evaluator.last_diagnostic
        assert diag["status_code"] is None
        assert diag["classified_error_type"] == "timeout"
        assert diag["timeout_seconds"] == 10.0


def test_gemini_diagnostic_classification_malformed_json():
    import httpx
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key="gemini-key")
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    mock_resp = httpx.Response(status_code=200, request=req, text="not-valid-json-response")

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all("Q", "A", "E", "C", ["doc-1"])
        assert status == "judge_failed"
        assert reason == "malformed_json"
        diag = evaluator.last_diagnostic
        assert diag["status_code"] == 200
        assert diag["classified_error_type"] == "malformed_json"


def test_gemini_diagnostic_secrets_redaction():
    import httpx
    secret_key = "GEMINI_SECRET_KEY_999999"
    evaluator = LLMJudgeEvaluator(provider="gemini", model="gemini-2.5-flash", api_key=secret_key)
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    body = json.dumps({
        "error": {
            "code": 401,
            "message": f"Invalid key {secret_key} in Authorization: Bearer {secret_key} and key={secret_key}",
            "type": "invalid_auth"
        }
    })
    mock_resp = httpx.Response(status_code=401, request=req, text=body)

    with patch("httpx.Client.post", return_value=mock_resp):
        corr, rel, claims, status, reason = evaluator.evaluate_all(
            question="CONFIDENTIAL QUESTION TEXT",
            answer="CONFIDENTIAL ANSWER TEXT",
            expected_answer="CONFIDENTIAL EXPECTED TEXT",
            context="CONFIDENTIAL CONTEXT DOCUMENT TEXT",
            retrieved_ids=["doc-1"]
        )
        assert status == "judge_failed"
        assert reason == "http_401_unauthorized"
        diag = evaluator.last_diagnostic
        assert diag["api_key_present"] is True
        
        diag_str = json.dumps(diag)
        assert secret_key not in diag_str
        assert "CONFIDENTIAL QUESTION TEXT" not in diag_str
        assert "CONFIDENTIAL ANSWER TEXT" not in diag_str
        assert "CONFIDENTIAL EXPECTED TEXT" not in diag_str
        assert "CONFIDENTIAL CONTEXT DOCUMENT TEXT" not in diag_str
        assert "***REDACTED***" in diag_str






