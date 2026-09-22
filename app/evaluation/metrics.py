import statistics
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

class CitationValidation(BaseModel):
    citation_text: str
    document_id: Optional[str] = None
    document_exists: bool
    document_was_retrieved: bool
    source_supports_claim: bool

class ClaimEvaluation(BaseModel):
    claim: str
    supported: bool
    supporting_source_ids: List[str] = Field(default_factory=list)
    citations: List[CitationValidation] = Field(default_factory=list)


def calculate_claim_faithfulness_score(claims: List[ClaimEvaluation]) -> float:
    """Calculates claim faithfulness as the ratio of supported claims to total claims.

    If no claims are present, returns 1.0.
    """
    if not claims:
        return 1.0
    supported_count = sum(1 for c in claims if c.supported)
    return float(supported_count / len(claims))


def calculate_citation_correctness_score(claims: List[ClaimEvaluation]) -> float:
    """Evaluates citation correctness based on 3 criteria per citation:

    1. document_exists
    2. document_was_retrieved
    3. source_supports_claim
    
    If no citations exist, returns 1.0.
    """
    total_criteria = 0
    correct_criteria = 0

    for claim in claims:
        for cit in claim.citations:
            total_criteria += 3
            if cit.document_exists:
                correct_criteria += 1
            if cit.document_was_retrieved:
                correct_criteria += 1
            if cit.source_supports_claim:
                correct_criteria += 1

    if total_criteria == 0:
        return 1.0

    return float(correct_criteria / total_criteria)


def evaluate_refusal_correctness(
    category: str,
    answer: str,
    relevance_threshold: float,
    top_score: float
) -> Tuple[bool, bool]:
    """Determines whether refusal behavior was expected and returned.

    refusal_correct = refusal_expected == refusal_returned
    """
    # System returns refusal if answer contains fallback indicator
    refusal_indicators = [
        "could not find information",
        "do not have information",
        "don't have information",
        "no information",
        "sorry, i cannot",
        "i could not"
    ]
    ans_lower = answer.lower()
    refusal_returned = any(ind in ans_lower for ind in refusal_indicators)

    # Refusal is expected if:
    # - question category is unanswerable or hallucination_check
    # - OR top retrieval score is below relevance threshold
    refusal_expected = (
        category in ("unanswerable", "hallucination_check")
        or top_score < relevance_threshold
    )

    refusal_correct = refusal_expected == refusal_returned
    return refusal_correct, refusal_returned


def aggregate_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall and category-specific average benchmarks."""
    
    def get_stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"mean": None, "median": None, "p95": None, "sample_count": 0}
        return {
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "sample_count": len(vals)
        }

    # Groups
    categories = ["direct_answer", "multi_part", "cross_document", "unanswerable", "hallucination_check"]
    cat_metrics = {}
    for cat in categories:
        cat_metrics[cat] = {
            "hit_at_5": [],
            "mrr_at_5": [],
            "correctness": [],
            "faithfulness": [],
            "relevance": [],
            "citation": [],
            "refusal_correct_count": 0,
            "refusal_total_count": 0,
            "latencies": []
        }

    overall = {
        "hit_at_5": [],
        "mrr_at_5": [],
        "correctness": [],
        "faithfulness": [],
        "relevance": [],
        "citation": [],
        "refusal_correct_count": 0,
        "refusal_total_count": 0,
        "latencies": []
    }

    judge_failures = 0

    for res in results:
        cat = res["category"]
        lat = res["latency"]["total_ms"]
        
        # Add to category
        if cat in cat_metrics:
            target_cat = cat_metrics[cat]
        else:
            continue
            
        target_cat["latencies"].append(lat)
        overall["latencies"].append(lat)

        # Retrieval metrics (applicable if not unanswerable/hallucination)
        if res.get("hit_at_5") is not None:
            target_cat["hit_at_5"].append(res["hit_at_5"])
            overall["hit_at_5"].append(res["hit_at_5"])
        if res.get("reciprocal_rank") is not None:
            target_cat["mrr_at_5"].append(res["reciprocal_rank"])
            overall["mrr_at_5"].append(res["reciprocal_rank"])

        # Refusal accuracy tracking
        target_cat["refusal_total_count"] += 1
        overall["refusal_total_count"] += 1
        if res.get("refusal_correct"):
            target_cat["refusal_correct_count"] += 1
            overall["refusal_correct_count"] += 1

        # Judge Quality Scores
        if res["judge_status"] == "judge_failed":
            judge_failures += 1
            continue

        if res.get("answer_correctness_score") is not None:
            target_cat["correctness"].append(res["answer_correctness_score"])
            overall["correctness"].append(res["answer_correctness_score"])

        if res.get("faithfulness_score") is not None:
            target_cat["faithfulness"].append(res["faithfulness_score"])
            overall["faithfulness"].append(res["faithfulness_score"])

        if res.get("context_relevance_score") is not None:
            target_cat["relevance"].append(res["context_relevance_score"])
            overall["relevance"].append(res["context_relevance_score"])

        if res.get("citation_correctness_score") is not None:
            target_cat["citation"].append(res["citation_correctness_score"])
            overall["citation"].append(res["citation_correctness_score"])

    # Aggregate summaries by category
    summary_by_cat = {}
    for cat in categories:
        target = cat_metrics[cat]
        ref_acc = (
            float(target["refusal_correct_count"] / target["refusal_total_count"])
            if target["refusal_total_count"] > 0
            else 1.0
        )
        summary_by_cat[cat] = {
            "retrieval": {
                "hit_rate_at_5": float(statistics.mean(target["hit_at_5"])) if target["hit_at_5"] else 0.0,
                "mrr_at_5": float(statistics.mean(target["mrr_at_5"])) if target["mrr_at_5"] else 0.0,
            },
            "generation": {
                "mean_correctness": float(statistics.mean(target["correctness"])) if target["correctness"] else None,
                "mean_faithfulness": float(statistics.mean(target["faithfulness"])) if target["faithfulness"] else None,
                "mean_context_relevance": float(statistics.mean(target["relevance"])) if target["relevance"] else None,
                "mean_citation_correctness": float(statistics.mean(target["citation"])) if target["citation"] else None,
            },
            "safety": {
                "refusal_accuracy": ref_acc,
                "failure_rate": 1.0 - ref_acc
            },
            "performance": get_stats(target["latencies"])
        }

    # Aggregate overall summary
    overall_ref_acc = (
        float(overall["refusal_correct_count"] / overall["refusal_total_count"])
        if overall["refusal_total_count"] > 0
        else 1.0
    )
    summary_overall = {
        "retrieval": {
            "hit_rate_at_5": float(statistics.mean(overall["hit_at_5"])) if overall["hit_at_5"] else 0.0,
            "mrr_at_5": float(statistics.mean(overall["mrr_at_5"])) if overall["mrr_at_5"] else 0.0,
        },
        "generation": {
            "mean_correctness": float(statistics.mean(overall["correctness"])) if overall["correctness"] else None,
            "mean_faithfulness": float(statistics.mean(overall["faithfulness"])) if overall["faithfulness"] else None,
            "mean_context_relevance": float(statistics.mean(overall["relevance"])) if overall["relevance"] else None,
            "mean_citation_correctness": float(statistics.mean(overall["citation"])) if overall["citation"] else None,
        },
        "safety": {
            "refusal_accuracy": overall_ref_acc,
            "failure_rate": 1.0 - overall_ref_acc
        },
        "performance": get_stats(overall["latencies"]),
        "judge_failures": judge_failures
    }

    return {
        "overall": summary_overall,
        "categories": summary_by_cat
    }
