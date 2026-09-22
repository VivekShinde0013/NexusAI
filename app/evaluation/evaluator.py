import json
import httpx
import re
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
from app.config import get_settings
from app.evaluation.metrics import ClaimEvaluation, CitationValidation


def calculate_keyword_match_score(
    answer: Optional[str], keywords: List[str]
) -> float:
    """Calculates the ratio of expected key terms present in the answer.

    This is a deterministic keyword-matching heuristic.
    It is NOT equivalent to semantic correctness or faithfulness.
    """
    if not answer or not keywords:
        return 0.0

    lower_answer = answer.lower()
    matches = 0
    for kw in keywords:
        if kw.lower() in lower_answer:
            matches += 1

    return float(matches / len(keywords))


def _classify_http_error(
    status_code: int, response_text: Optional[str] = None
) -> str:
    """Safely classifies HTTP status codes into distinct error reason categories without exposing sensitive content."""
    if status_code == 401:
        return "http_401_unauthorized"
    elif status_code == 403:
        return "http_403_forbidden"
    elif status_code == 429:
        if response_text:
            lowered = response_text.lower()
            quota_keywords = (
                "quota",
                "exhausted",
                "credit",
                "resource_exhausted",
                "insufficient_quota",
                "quota_exceeded",
            )
            if any(kw in lowered for kw in quota_keywords):
                return "http_429_quota_exhausted"
        return "http_429_rate_limited"
    elif status_code == 400:
        return "http_400_bad_request"
    elif status_code == 404:
        return "http_404_not_found"
    elif 500 <= status_code <= 599:
        return "http_5xx_server_error"
    else:
        return "http_error_other"


def _sanitize_text(text: Optional[str], api_key: Optional[str] = None) -> Optional[str]:
    """Sanitizes text by stripping out API keys, Authorization headers, and tokens."""
    if text is None:
        return None
    sanitized = str(text)
    if api_key and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "***REDACTED***")

    sanitized = re.sub(
        r"Bearer\s+[A-Za-z0-9_\-\.]+",
        "Bearer ***REDACTED***",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(key|api_key|secret|token|authorization)=[^&\s]+",
        r"\1=***REDACTED***",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"AIzaSy[A-Za-z0-9_\-]{33}", "***REDACTED***", sanitized)
    sanitized = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "***REDACTED***", sanitized)
    return sanitized


def _extract_safe_error_details(
    raw_response: Optional[str], api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Extracts type, code, and sanitized message from response body or exception text."""
    if not raw_response:
        return None

    err_type = None
    err_code = None
    err_msg = None

    try:
        data = json.loads(raw_response)
        if isinstance(data, dict) and "error" in data:
            err_obj = data["error"]
            if isinstance(err_obj, dict):
                err_type = err_obj.get("type") or err_obj.get("status")
                err_code = err_obj.get("code")
                err_msg = err_obj.get("message")
            elif isinstance(err_obj, str):
                err_msg = err_obj
    except Exception:
        err_msg = raw_response[:200]

    return {
        "type": _sanitize_text(str(err_type) if err_type is not None else None, api_key),
        "code": err_code if isinstance(err_code, (int, str)) else None,
        "message": _sanitize_text(str(err_msg) if err_msg is not None else None, api_key),
    }


class LLMJudgeEvaluator:

    _gemini_lock = threading.Lock()
    _gemini_last_request_time = 0.0

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        rpm_limit: Optional[int] = None,
    ):
        if provider not in ("mock", "openai", "gemini"):
            raise ValueError(f"Unsupported LLM Judge provider: '{provider}'")
        self.provider = provider
        self.model = model
        
        settings = get_settings()
        if api_key is not None:
            self.api_key = api_key
        else:
            if provider == "gemini":
                self.api_key = (
                    settings.EVAL_QUALITY_API_KEY
                    or settings.GEMINI_API_KEY
                    or settings.LLM_API_KEY
                )
            else:
                self.api_key = settings.EVAL_QUALITY_API_KEY or settings.LLM_API_KEY

        if provider == "gemini":
            self.api_url = settings.GEMINI_API_URL
        else:
            self.api_url = settings.LLM_API_URL

        self.rpm_limit = (
            rpm_limit if rpm_limit is not None else settings.EVAL_QUALITY_RPM_LIMIT
        )
        self.last_diagnostic: Optional[Dict[str, Any]] = None

    def _format_endpoint_url(self) -> str:
        if not self.api_url:
            return ""
        try:
            parsed = urllib.parse.urlparse(self.api_url)
            if parsed.netloc:
                return f"{parsed.netloc}{parsed.path}"
            return self.api_url.split("?")[0]
        except Exception:
            return self.api_url.split("?")[0]

    def _record_diagnostic(
        self,
        status_code: Optional[int],
        classified_error_type: str,
        latency_ms: float,
        error_details: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 10.0,
    ):
        self.last_diagnostic = {
            "status_code": status_code,
            "classified_error_type": classified_error_type,
            "provider": self.provider,
            "model": self.model,
            "endpoint_url": self._format_endpoint_url(),
            "request_latency_ms": round(latency_ms, 2),
            "timeout_seconds": timeout_seconds,
            "api_key_present": bool(self.api_key),
            "error_details": error_details,
        }

    def _throttle_if_needed(self):
        """Throttles requests for Gemini provider to respect configured RPM limit."""
        if self.provider != "gemini" or self.rpm_limit <= 0:
            return

        interval = 60.0 / self.rpm_limit
        with LLMJudgeEvaluator._gemini_lock:
            now = time.perf_counter()
            elapsed = now - LLMJudgeEvaluator._gemini_last_request_time
            if elapsed < interval:
                wait_time = interval - elapsed
                time.sleep(wait_time)
            LLMJudgeEvaluator._gemini_last_request_time = time.perf_counter()

    def evaluate_all(
        self,
        question: str,
        answer: str,
        expected_answer: Optional[str],
        context: str,
        retrieved_ids: List[str],
    ) -> Tuple[Optional[float], Optional[float], List[ClaimEvaluation], str, Optional[str]]:
        """Evaluates correctness, context relevance, and claims/citations in a single LLM Judge API call.

        Returns: (correctness_score, context_relevance_score, claims, status, failure_reason)
        """
        if self.provider == "mock":
            # 1. Correctness mock
            if not expected_answer:
                corr_score = 1.0
            else:
                kw = expected_answer.split()
                corr_score = float(calculate_keyword_match_score(answer, kw))

            # 2. Relevance mock
            kw_q = question.split()
            rel_score = float(calculate_keyword_match_score(context, kw_q))

            # 3. Claims & Citations mock
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
            claims = []
            for idx, sentence in enumerate(sentences):
                supported = False
                supporting = []
                for rid in retrieved_ids:
                    if "leave" in sentence.lower() and rid == "11111111-1111-1111-1111-111111111111":
                        supported = True
                        supporting.append(rid)
                    elif "hours" in sentence.lower() and rid == "11111111-1111-1111-1111-111111111111":
                        supported = True
                        supporting.append(rid)
                    elif "hmo" in sentence.lower() and rid == "22222222-2222-2222-2222-222222222222":
                        supported = True
                        supporting.append(rid)
                    elif "gym" in sentence.lower() and rid == "22222222-2222-2222-2222-222222222222":
                        supported = True
                        supporting.append(rid)
                    elif "password" in sentence.lower() and rid == "33333333-3333-3333-3333-333333333333":
                        supported = True
                        supporting.append(rid)
                    elif "phishing" in sentence.lower() and rid == "33333333-3333-3333-3333-333333333333":
                        supported = True
                        supporting.append(rid)

                if not supported and len(context) > 0:
                    clean_sent = re.sub(r'[^\w\s]', '', sentence.lower())
                    clean_context = re.sub(r'[^\w\s]', '', context.lower())
                    overlap_words = [w for w in clean_sent.split() if w in clean_context]
                    if len(overlap_words) > 3:
                        supported = True
                        if retrieved_ids:
                            supporting.append(retrieved_ids[0])

                citations = []
                matches = re.findall(r'\[\d+\]', sentence)
                for match in matches:
                    try:
                        cit_idx = int(match.strip("[]")) - 1
                        doc_id = retrieved_ids[cit_idx] if cit_idx < len(retrieved_ids) else "doc-nonexistent"
                    except Exception:
                        doc_id = "doc-nonexistent"

                    doc_exists = doc_id in ("11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333")
                    retrieved = doc_id in retrieved_ids
                    supports = doc_id in supporting if supported else False

                    citations.append(
                        CitationValidation(
                            citation_text=match,
                            document_id=doc_id,
                            document_exists=doc_exists,
                            document_was_retrieved=retrieved,
                            source_supports_claim=supports
                        )
                    )

                claims.append(
                    ClaimEvaluation(
                        claim=sentence,
                        supported=supported,
                        supporting_source_ids=supporting,
                        citations=citations
                    )
                )

            self._record_diagnostic(
                status_code=200,
                classified_error_type="none",
                latency_ms=0.0,
                error_details=None,
            )
            return corr_score, rel_score, claims, "success", None

        if self.provider in ("openai", "gemini"):
            if not self.api_key:
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="missing_required_field",
                    latency_ms=0.0,
                    error_details={"type": "ConfigurationError", "code": None, "message": "API key missing"},
                )
                return None, None, [], "judge_failed", "missing_required_field"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            prompt = (
                "You are an objective AI evaluation judge. Evaluate the generated RAG system answer across three aspects:\n\n"
                "1. Correctness: Rate the factual correctness and completeness of the generated answer compared to the expected answer on a scale from 1 to 5 (where 1 is completely incorrect/missing and 5 is perfectly correct and complete).\n"
                "2. Context Relevance: Rate how relevant the retrieved context is to answering the user question on a scale from 1 to 5 (1 is completely irrelevant, 5 is highly relevant).\n"
                "3. Claims & Citations: Extract key factual claims from the generated answer. For each claim, evaluate:\n"
                "   a. Whether it is supported by the retrieved context.\n"
                "   b. List supporting_source_ids matching the provided retrieved document IDs (do not invent document IDs).\n"
                "   c. List any citations present in the claim (e.g. '[1]', '[doc.txt]'), rating whether the cited document exists, whether it was in the retrieved set, and whether it supports the claim.\n\n"
                f"Question: {question}\n"
                f"Generated Answer: {answer}\n"
                f"Expected Answer: {expected_answer}\n"
                f"Context: {context}\n"
                f"Retrieved Document IDs: {retrieved_ids}\n\n"
                "Return JSON matching:\n"
                "{\n"
                '  "correctness_score": float,\n'
                '  "context_relevance_score": float,\n'
                '  "claims": [\n'
                "    {\n"
                '      "claim": "string",\n'
                '      "supported": boolean,\n'
                '      "supporting_source_ids": ["string"],\n'
                '      "citations": [\n'
                "        {\n"
                '          "citation_text": "string",\n'
                '          "document_id": "string",\n'
                '          "document_exists": boolean,\n'
                '          "document_was_retrieved": boolean,\n'
                '          "source_supports_claim": boolean\n'
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}"
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            start_time = time.perf_counter()
            try:
                self._throttle_if_needed()
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(self.api_url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code != 200:
                    resp_text = resp.text
                    classified = _classify_http_error(resp.status_code, resp_text)
                    details = _extract_safe_error_details(resp_text, self.api_key)
                    self._record_diagnostic(
                        status_code=resp.status_code,
                        classified_error_type=classified,
                        latency_ms=latency_ms,
                        error_details=details,
                    )
                    return None, None, [], "judge_failed", classified

                data = resp.json()
                content_str = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                parsed = json.loads(content_str)

                if not all(k in parsed for k in ("correctness_score", "context_relevance_score", "claims")):
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="missing_required_field",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Missing required field in JSON payload"},
                    )
                    return None, None, [], "judge_failed", "missing_required_field"

                corr_val = parsed["correctness_score"]
                rel_val = parsed["context_relevance_score"]
                claims_raw = parsed["claims"]

                if not isinstance(corr_val, (int, float)) or not isinstance(rel_val, (int, float)):
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Score is not numeric"},
                    )
                    return None, None, [], "judge_failed", "invalid_score"

                if not isinstance(claims_raw, list):
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="missing_required_field",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Claims is not a list"},
                    )
                    return None, None, [], "judge_failed", "missing_required_field"

                if 1.0 <= corr_val <= 5.0:
                    norm_corr = float((corr_val - 1.0) / 4.0)
                elif 0.0 <= corr_val <= 1.0:
                    norm_corr = float(corr_val)
                else:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Correctness score out of range"},
                    )
                    return None, None, [], "judge_failed", "invalid_score"

                if 1.0 <= rel_val <= 5.0:
                    norm_rel = float((rel_val - 1.0) / 4.0)
                elif 0.0 <= rel_val <= 1.0:
                    norm_rel = float(rel_val)
                else:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Relevance score out of range"},
                    )
                    return None, None, [], "judge_failed", "invalid_score"

                claims = []
                for c in claims_raw:
                    if not isinstance(c, dict) or "claim" not in c or "supported" not in c:
                        self._record_diagnostic(
                            status_code=200,
                            classified_error_type="missing_required_field",
                            latency_ms=latency_ms,
                            error_details={"type": "ValidationError", "code": None, "message": "Claim item missing claim or supported field"},
                        )
                        return None, None, [], "judge_failed", "missing_required_field"
                    if not isinstance(c["claim"], str) or not isinstance(c["supported"], bool):
                        self._record_diagnostic(
                            status_code=200,
                            classified_error_type="missing_required_field",
                            latency_ms=latency_ms,
                            error_details={"type": "ValidationError", "code": None, "message": "Claim item invalid field types"},
                        )
                        return None, None, [], "judge_failed", "missing_required_field"

                    cits = []
                    for cit in c.get("citations", []):
                        if not isinstance(cit, dict):
                            self._record_diagnostic(
                                status_code=200,
                                classified_error_type="missing_required_field",
                                latency_ms=latency_ms,
                                error_details={"type": "ValidationError", "code": None, "message": "Citation item invalid type"},
                            )
                            return None, None, [], "judge_failed", "missing_required_field"
                        if any(k not in cit for k in ("citation_text", "document_exists", "document_was_retrieved", "source_supports_claim")):
                            self._record_diagnostic(
                                status_code=200,
                                classified_error_type="missing_required_field",
                                latency_ms=latency_ms,
                                error_details={"type": "ValidationError", "code": None, "message": "Citation item missing required field"},
                            )
                            return None, None, [], "judge_failed", "missing_required_field"
                        cits.append(
                            CitationValidation(
                                citation_text=str(cit["citation_text"]),
                                document_id=cit.get("document_id"),
                                document_exists=bool(cit["document_exists"]),
                                document_was_retrieved=bool(cit["document_was_retrieved"]),
                                source_supports_claim=bool(cit["source_supports_claim"])
                            )
                        )

                    claims.append(
                        ClaimEvaluation(
                            claim=c["claim"],
                            supported=c["supported"],
                            supporting_source_ids=c.get("supporting_source_ids", []),
                            citations=cits
                        )
                    )

                self._record_diagnostic(
                    status_code=200,
                    classified_error_type="none",
                    latency_ms=latency_ms,
                    error_details=None,
                )
                return norm_corr, norm_rel, claims, "success", None
            except json.JSONDecodeError:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(
                    status_code=200,
                    classified_error_type="malformed_json",
                    latency_ms=latency_ms,
                    error_details={"type": "JSONDecodeError", "code": None, "message": "Failed to parse JSON content from judge response"},
                )
                return None, None, [], "judge_failed", "malformed_json"
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="timeout",
                    latency_ms=latency_ms,
                    error_details={"type": "TimeoutException", "code": None, "message": "Judge request timed out"},
                )
                return None, None, [], "judge_failed", "timeout"
            except httpx.HTTPStatusError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = e.response.status_code if e.response is not None else 0
                resp_text = e.response.text if e.response is not None else ""
                classified = _classify_http_error(status_code, resp_text)
                details = _extract_safe_error_details(resp_text, self.api_key)
                self._record_diagnostic(
                    status_code=status_code,
                    classified_error_type=classified,
                    latency_ms=latency_ms,
                    error_details=details,
                )
                return None, None, [], "judge_failed", classified
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                details = _extract_safe_error_details(str(e), self.api_key)
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="connection_error",
                    latency_ms=latency_ms,
                    error_details=details,
                )
                return None, None, [], "judge_failed", "connection_error"

        self._record_diagnostic(
            status_code=None,
            classified_error_type="unsupported_provider",
            latency_ms=0.0,
            error_details={"type": "ConfigurationError", "code": None, "message": f"Unsupported provider: {self.provider}"},
        )
        return None, None, [], "judge_failed", "unsupported_provider"

    def evaluate_correctness(
        self, question: str, answer: str, expected_answer: Optional[str]
    ) -> Tuple[Optional[float], str, Optional[str]]:
        """Evaluates semantic correctness of the answer compared to reference.

        Returns: (score, status, failure_reason)
        """
        if self.provider == "mock":
            if not expected_answer:
                self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=0.0)
                return 1.0, "success", None
            kw = expected_answer.split()
            score = calculate_keyword_match_score(answer, kw)
            self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=0.0)
            return float(score), "success", None

        if self.provider in ("openai", "gemini"):
            if not self.api_key:
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="missing_required_field",
                    latency_ms=0.0,
                    error_details={"type": "ConfigurationError", "code": None, "message": "API key missing"},
                )
                return None, "judge_failed", "missing_required_field"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            prompt = (
                "You are an objective AI evaluation judge. Rate the correctness of the generated answer compared to the ground truth expected answer on a scale from 1 to 5 (where 1 is completely incorrect and 5 is perfectly correct).\n"
                f"Question: {question}\n"
                f"Generated Answer: {answer}\n"
                f"Expected Answer: {expected_answer}\n"
                "Return JSON matching: {\"score\": float}"
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            start_time = time.perf_counter()
            try:
                self._throttle_if_needed()
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(self.api_url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code != 200:
                    resp_text = resp.text
                    classified = _classify_http_error(resp.status_code, resp_text)
                    details = _extract_safe_error_details(resp_text, self.api_key)
                    self._record_diagnostic(
                        status_code=resp.status_code,
                        classified_error_type=classified,
                        latency_ms=latency_ms,
                        error_details=details,
                    )
                    return None, "judge_failed", classified

                data = resp.json()
                score_str = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                parsed = json.loads(score_str)
                if "score" not in parsed:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="missing_required_field",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Missing score field"},
                    )
                    return None, "judge_failed", "missing_required_field"
                
                score_val = parsed["score"]
                if not isinstance(score_val, (int, float)):
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Score is not numeric"},
                    )
                    return None, "judge_failed", "invalid_score"

                if 1.0 <= score_val <= 5.0:
                    normalized = float((score_val - 1.0) / 4.0)
                elif 0.0 <= score_val <= 1.0:
                    normalized = float(score_val)
                else:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Score out of range"},
                    )
                    return None, "judge_failed", "invalid_score"
                
                self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=latency_ms, error_details=None)
                return normalized, "success", None
            except json.JSONDecodeError:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=200, classified_error_type="malformed_json", latency_ms=latency_ms, error_details={"type": "JSONDecodeError", "code": None, "message": "Failed to parse JSON content"})
                return None, "judge_failed", "malformed_json"
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=None, classified_error_type="timeout", latency_ms=latency_ms, error_details={"type": "TimeoutException", "code": None, "message": "Judge request timed out"})
                return None, "judge_failed", "timeout"
            except httpx.HTTPStatusError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = e.response.status_code if e.response is not None else 0
                resp_text = e.response.text if e.response is not None else ""
                classified = _classify_http_error(status_code, resp_text)
                details = _extract_safe_error_details(resp_text, self.api_key)
                self._record_diagnostic(status_code=status_code, classified_error_type=classified, latency_ms=latency_ms, error_details=details)
                return None, "judge_failed", classified
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                details = _extract_safe_error_details(str(e), self.api_key)
                self._record_diagnostic(status_code=None, classified_error_type="connection_error", latency_ms=latency_ms, error_details=details)
                return None, "judge_failed", "connection_error"

        self._record_diagnostic(status_code=None, classified_error_type="unsupported_provider", latency_ms=0.0)
        return None, "judge_failed", "unsupported_provider"

    def evaluate_context_relevance(
        self, question: str, context: str
    ) -> Tuple[Optional[float], str, Optional[str]]:
        """Evaluates whether context is relevant to the question."""
        if self.provider == "mock":
            kw = question.split()
            score = calculate_keyword_match_score(context, kw)
            self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=0.0)
            return float(score), "success", None

        if self.provider in ("openai", "gemini"):
            if not self.api_key:
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="missing_required_field",
                    latency_ms=0.0,
                    error_details={"type": "ConfigurationError", "code": None, "message": "API key missing"},
                )
                return None, "judge_failed", "missing_required_field"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            prompt = (
                "Rate how relevant the retrieved context is to answering the user question on a scale from 1 to 5 (1 is completely irrelevant, 5 is highly relevant).\n"
                f"Question: {question}\n"
                f"Context: {context}\n"
                "Return JSON matching: {\"score\": float}"
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            start_time = time.perf_counter()
            try:
                self._throttle_if_needed()
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(self.api_url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code != 200:
                    resp_text = resp.text
                    classified = _classify_http_error(resp.status_code, resp_text)
                    details = _extract_safe_error_details(resp_text, self.api_key)
                    self._record_diagnostic(
                        status_code=resp.status_code,
                        classified_error_type=classified,
                        latency_ms=latency_ms,
                        error_details=details,
                    )
                    return None, "judge_failed", classified

                data = resp.json()
                score_str = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                parsed = json.loads(score_str)
                if "score" not in parsed:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="missing_required_field",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Missing score field"},
                    )
                    return None, "judge_failed", "missing_required_field"
                
                score_val = parsed["score"]
                if not isinstance(score_val, (int, float)):
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Score is not numeric"},
                    )
                    return None, "judge_failed", "invalid_score"

                if 1.0 <= score_val <= 5.0:
                    normalized = float((score_val - 1.0) / 4.0)
                elif 0.0 <= score_val <= 1.0:
                    normalized = float(score_val)
                else:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="invalid_score",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Score out of range"},
                    )
                    return None, "judge_failed", "invalid_score"

                self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=latency_ms, error_details=None)
                return normalized, "success", None
            except json.JSONDecodeError:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=200, classified_error_type="malformed_json", latency_ms=latency_ms, error_details={"type": "JSONDecodeError", "code": None, "message": "Failed to parse JSON content"})
                return None, "judge_failed", "malformed_json"
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=None, classified_error_type="timeout", latency_ms=latency_ms, error_details={"type": "TimeoutException", "code": None, "message": "Judge request timed out"})
                return None, "judge_failed", "timeout"
            except httpx.HTTPStatusError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = e.response.status_code if e.response is not None else 0
                resp_text = e.response.text if e.response is not None else ""
                classified = _classify_http_error(status_code, resp_text)
                details = _extract_safe_error_details(resp_text, self.api_key)
                self._record_diagnostic(status_code=status_code, classified_error_type=classified, latency_ms=latency_ms, error_details=details)
                return None, "judge_failed", classified
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                details = _extract_safe_error_details(str(e), self.api_key)
                self._record_diagnostic(status_code=None, classified_error_type="connection_error", latency_ms=latency_ms, error_details=details)
                return None, "judge_failed", "connection_error"

        self._record_diagnostic(status_code=None, classified_error_type="unsupported_provider", latency_ms=0.0)
        return None, "judge_failed", "unsupported_provider"

    def evaluate_claims_and_citations(
        self, question: str, answer: str, context: str, retrieved_ids: List[str]
    ) -> Tuple[List[ClaimEvaluation], str, Optional[str]]:
        """Extracts claims and citation mappings from the generated answer and validates them.

        Returns: (claims, status, failure_reason)
        """
        if self.provider == "mock":
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
            claims = []
            
            for idx, sentence in enumerate(sentences):
                supported = False
                supporting = []
                
                for rid in retrieved_ids:
                    if "leave" in sentence.lower() and rid == "11111111-1111-1111-1111-111111111111":
                        supported = True
                        supporting.append(rid)
                    elif "hours" in sentence.lower() and rid == "11111111-1111-1111-1111-111111111111":
                        supported = True
                        supporting.append(rid)
                    elif "hmo" in sentence.lower() and rid == "22222222-2222-2222-2222-222222222222":
                        supported = True
                        supporting.append(rid)
                    elif "gym" in sentence.lower() and rid == "22222222-2222-2222-2222-222222222222":
                        supported = True
                        supporting.append(rid)
                    elif "password" in sentence.lower() and rid == "33333333-3333-3333-3333-333333333333":
                        supported = True
                        supporting.append(rid)
                    elif "phishing" in sentence.lower() and rid == "33333333-3333-3333-3333-333333333333":
                        supported = True
                        supporting.append(rid)

                if not supported and len(context) > 0:
                    clean_sent = re.sub(r'[^\w\s]', '', sentence.lower())
                    clean_context = re.sub(r'[^\w\s]', '', context.lower())
                    overlap_words = [w for w in clean_sent.split() if w in clean_context]
                    if len(overlap_words) > 3:
                        supported = True
                        if retrieved_ids:
                            supporting.append(retrieved_ids[0])

                citations = []
                matches = re.findall(r'\[\d+\]', sentence)
                for match in matches:
                    try:
                        cit_idx = int(match.strip("[]")) - 1
                        doc_id = retrieved_ids[cit_idx] if cit_idx < len(retrieved_ids) else "doc-nonexistent"
                    except Exception:
                        doc_id = "doc-nonexistent"
                    
                    doc_exists = doc_id in ("11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333")
                    retrieved = doc_id in retrieved_ids
                    supports = doc_id in supporting if supported else False
                    
                    citations.append(
                        CitationValidation(
                            citation_text=match,
                            document_id=doc_id,
                            document_exists=doc_exists,
                            document_was_retrieved=retrieved,
                            source_supports_claim=supports
                        )
                    )

                claims.append(
                    ClaimEvaluation(
                        claim=sentence,
                        supported=supported,
                        supporting_source_ids=supporting,
                        citations=citations
                    )
                )

            self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=0.0)
            return claims, "success", None

        if self.provider in ("openai", "gemini"):
            if not self.api_key:
                self._record_diagnostic(
                    status_code=None,
                    classified_error_type="missing_required_field",
                    latency_ms=0.0,
                    error_details={"type": "ConfigurationError", "code": None, "message": "API key missing"},
                )
                return [], "judge_failed", "missing_required_field"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            prompt = (
                "You are an AI evaluation judge. Extract the list of key factual claims from the generated answer. "
                "For each claim, evaluate: \n"
                "1. If it is supported by the retrieved context.\n"
                "2. What are the supporting_source_ids (matching the document IDs in the context).\n"
                "3. Any citations present in the claim (e.g. '[1]', '[benefits_policy.txt]'). Rate whether the cited document exists, "
                "whether it was in the retrieved set, and whether the cited document actually supports the claim.\n"
                f"Question: {question}\n"
                f"Generated Answer: {answer}\n"
                f"Context: {context}\n"
                f"Retrieved Document IDs: {retrieved_ids}\n"
                "Return JSON matching:\n"
                "{\n"
                "  \"claims\": [\n"
                "    {\n"
                "      \"claim\": \"string\",\n"
                "      \"supported\": boolean,\n"
                "      \"supporting_source_ids\": [\"string\"],\n"
                "      \"citations\": [\n"
                "        {\n"
                "          \"citation_text\": \"string\",\n"
                "          \"document_id\": \"string\",\n"
                "          \"document_exists\": boolean,\n"
                "          \"document_was_retrieved\": boolean,\n"
                "          \"source_supports_claim\": boolean\n"
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}"
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            start_time = time.perf_counter()
            try:
                self._throttle_if_needed()
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(self.api_url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code != 200:
                    resp_text = resp.text
                    classified = _classify_http_error(resp.status_code, resp_text)
                    details = _extract_safe_error_details(resp_text, self.api_key)
                    self._record_diagnostic(
                        status_code=resp.status_code,
                        classified_error_type=classified,
                        latency_ms=latency_ms,
                        error_details=details,
                    )
                    return [], "judge_failed", classified

                data = resp.json()
                score_str = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                parsed = json.loads(score_str)
                if "claims" not in parsed:
                    self._record_diagnostic(
                        status_code=200,
                        classified_error_type="missing_required_field",
                        latency_ms=latency_ms,
                        error_details={"type": "ValidationError", "code": None, "message": "Missing claims field"},
                    )
                    return [], "judge_failed", "missing_required_field"

                claims = []
                for c in parsed["claims"]:
                    if "claim" not in c or "supported" not in c:
                        self._record_diagnostic(
                            status_code=200,
                            classified_error_type="missing_required_field",
                            latency_ms=latency_ms,
                            error_details={"type": "ValidationError", "code": None, "message": "Missing claim or supported field"},
                        )
                        return [], "judge_failed", "missing_required_field"
                    
                    cits = []
                    for cit in c.get("citations", []):
                        if any(k not in cit for k in ("citation_text", "document_exists", "document_was_retrieved", "source_supports_claim")):
                            self._record_diagnostic(
                                status_code=200,
                                classified_error_type="missing_required_field",
                                latency_ms=latency_ms,
                                error_details={"type": "ValidationError", "code": None, "message": "Citation item missing field"},
                            )
                            return [], "judge_failed", "missing_required_field"
                        cits.append(
                            CitationValidation(
                                citation_text=cit["citation_text"],
                                document_id=cit.get("document_id"),
                                document_exists=cit["document_exists"],
                                document_was_retrieved=cit["document_was_retrieved"],
                                source_supports_claim=cit["source_supports_claim"]
                            )
                        )
                    
                    claims.append(
                        ClaimEvaluation(
                            claim=c["claim"],
                            supported=c["supported"],
                            supporting_source_ids=c.get("supporting_source_ids", []),
                            citations=cits
                        )
                    )

                self._record_diagnostic(status_code=200, classified_error_type="none", latency_ms=latency_ms, error_details=None)
                return claims, "success", None
            except json.JSONDecodeError:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=200, classified_error_type="malformed_json", latency_ms=latency_ms, error_details={"type": "JSONDecodeError", "code": None, "message": "Failed to parse JSON content"})
                return [], "judge_failed", "malformed_json"
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._record_diagnostic(status_code=None, classified_error_type="timeout", latency_ms=latency_ms, error_details={"type": "TimeoutException", "code": None, "message": "Judge request timed out"})
                return [], "judge_failed", "timeout"
            except httpx.HTTPStatusError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = e.response.status_code if e.response is not None else 0
                resp_text = e.response.text if e.response is not None else ""
                classified = _classify_http_error(status_code, resp_text)
                details = _extract_safe_error_details(resp_text, self.api_key)
                self._record_diagnostic(status_code=status_code, classified_error_type=classified, latency_ms=latency_ms, error_details=details)
                return [], "judge_failed", classified
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                details = _extract_safe_error_details(str(e), self.api_key)
                self._record_diagnostic(status_code=None, classified_error_type="connection_error", latency_ms=latency_ms, error_details=details)
                return [], "judge_failed", "connection_error"

        self._record_diagnostic(status_code=None, classified_error_type="unsupported_provider", latency_ms=0.0)
        return [], "judge_failed", "unsupported_provider"

