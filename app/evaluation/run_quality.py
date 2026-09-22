import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings, Settings, PROJECT_ROOT
from app.services.vector_store import get_vector_store
from app.services.embedding_service import get_embedding_service
from app.services.chunking_service import chunk_document
from app.services.document_extractor import extract_document
from app.services.indexing_service import index_chunks
from app.services import metadata_service
from app.services.context_builder import build_context
from app.services.llm_service import get_llm_service
from app.evaluation.evaluator import LLMJudgeEvaluator
from app.evaluation.metrics import (
    ClaimEvaluation,
    CitationValidation,
    calculate_claim_faithfulness_score,
    calculate_citation_correctness_score,
    evaluate_refusal_correctness,
    aggregate_quality_metrics
)
from app.evaluation.run import SYNTHETIC_CORPUS

def run_quality_evaluation(
    provider: str,
    model: str,
    dataset: Dict[str, Any],
    api_key: Optional[str] = None,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Runs quality evaluation of generated answers across the 35-question dataset.

    Executes inside an isolated environment and saves traces safely.
    """
    temp_dir = tempfile.mkdtemp()
    temp_raw = Path(temp_dir) / "raw"
    temp_vectors = Path(temp_dir) / "vectors"
    temp_raw.mkdir(parents=True)
    temp_vectors.mkdir(parents=True)

    # Isolated Env
    os.environ["UPLOAD_DIR"] = str(temp_raw)
    os.environ["VECTOR_STORE_DIR"] = str(temp_vectors)
    os.environ["CHUNK_SIZE"] = "500"
    os.environ["CHUNK_OVERLAP"] = "50"
    os.environ["EVAL_QUALITY_TEMPERATURE"] = str(temperature)

    get_settings.cache_clear()
    import app.services.llm_service as llm_service_module
    llm_service_module._llm_service_instance = None
    settings = get_settings()

    store = get_vector_store()
    store.clear()

    # 1. Index Synthetic Corpus
    for doc_id, doc_info in SYNTHETIC_CORPUS.items():
        raw_file_path = temp_raw / doc_info["filename"]
        with open(raw_file_path, "w", encoding="utf-8") as f:
            f.write(doc_info["content"])

        metadata_service.create_metadata(
            document_id=doc_id,
            original_filename=doc_info["filename"],
            stored_filename=doc_info["filename"],
            file_type="txt",
            file_size=len(doc_info["content"]),
        )

        extracted_doc = extract_document(
            document_id=doc_id,
            original_filename=doc_info["filename"],
            file_path=raw_file_path,
            file_type="txt",
        )
        chunks = chunk_document(
            extracted_doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        index_chunks(doc_id, chunks)

    # 2. Run Test Questions and Evaluate
    llm_service = get_llm_service()
    evaluator = LLMJudgeEvaluator(provider=provider, model=model, api_key=api_key)

    per_question_results = []

    for q in dataset["questions"]:
        question_id = q["id"]
        question_text = q["question"]
        expected_ans = q.get("expected_answer")
        expected_docs = q.get("expected_source_documents", [])
        category = q["category"]

        # Time total pipeline latency
        start_pipeline = time.perf_counter()
        
        # Generation step
        embedding_service = get_embedding_service()
        query_vector = embedding_service.generate_embeddings([question_text])
        search_results = store.search(query_vector, top_k=5)

        chunks_only = [item[0] for item in search_results]
        retrieved_source_docs = list(dict.fromkeys(chunk.document_id for chunk in chunks_only))

        highest_score = max(score for _, score in search_results) if search_results else 0.0

        ans_text = ""
        context_text = ""
        hit_at_5 = None
        mrr_at_5 = None

        if category not in ("unanswerable", "hallucination_check"):
            # Compute Hit@5 and MRR@5 for retrieval
            from app.evaluation.run import calculate_hit_at_k, calculate_mrr_at_k
            hit_at_5 = calculate_hit_at_k(chunks_only, expected_docs, k=5)
            mrr_at_5 = calculate_mrr_at_k(chunks_only, expected_docs, k=5)

        if highest_score < settings.LLM_RELEVANCE_THRESHOLD:
            ans_text = "I could not find information about this in the provided documents."
        else:
            seen_chunks = {}
            for chunk, score in search_results:
                if score < settings.LLM_RELEVANCE_THRESHOLD:
                    continue
                cid = chunk.chunk_id
                if cid not in seen_chunks or score > seen_chunks[cid][1]:
                    seen_chunks[cid] = (chunk, score)
            filtered_chunks = [item[0] for item in seen_chunks.values()]
            context_text = build_context(filtered_chunks)
            ans_text = llm_service.generate_answer(question_text, context_text)

        pipeline_lat = (time.perf_counter() - start_pipeline) * 1000.0

        # LLM Quality Judge steps
        start_judge = time.perf_counter()
        correctness_score = None
        faithfulness_score = None
        relevance_score = None
        citation_score = None
        claims_eval = []
        judge_status = "success"
        judge_reason = None

        # Determine refusal behavior
        ref_correct, ref_returned = evaluate_refusal_correctness(
            category, ans_text, settings.LLM_RELEVANCE_THRESHOLD, highest_score
        )
        ref_expected = category in ("unanswerable", "hallucination_check") or highest_score < settings.LLM_RELEVANCE_THRESHOLD

        # If answer was a refusal, judge skipped (quality metrics are None, but status = success)
        if ref_returned:
            judge_status = "success"
        else:
            corr, rel, claims, stat, reason = evaluator.evaluate_all(
                question_text, ans_text, expected_ans, context_text, retrieved_source_docs
            )
            if stat == "judge_failed":
                judge_status = "judge_failed"
                judge_reason = reason
            else:
                correctness_score = corr
                relevance_score = rel
                claims_eval = claims
                faithfulness_score = calculate_claim_faithfulness_score(claims)
                citation_score = calculate_citation_correctness_score(claims)

        judge_lat = (time.perf_counter() - start_judge) * 1000.0
        total_eval_lat = pipeline_lat + judge_lat

        # Build detailed claim map for citation correctness serialization
        claims_serialized = []
        for c in claims_eval:
            cits = []
            for cit in c.citations:
                cits.append({
                    "citation_text": cit.citation_text,
                    "document_id": cit.document_id,
                    "document_exists": cit.document_exists,
                    "document_was_retrieved": cit.document_was_retrieved,
                    "source_supports_claim": cit.source_supports_claim
                })
            claims_serialized.append({
                "claim": c.claim,
                "supported": c.supported,
                "supporting_source_ids": c.supporting_source_ids,
                "citations": cits
            })

        per_question_results.append({
            "question_id": question_id,
            "category": category,
            "question": question_text,
            "expected_answer": expected_ans,
            "generated_answer": ans_text,
            "retrieved_sources": retrieved_source_docs,
            "hit_at_5": hit_at_5,
            "reciprocal_rank": mrr_at_5,
            "claims": claims_serialized,
            "faithfulness_score": faithfulness_score,
            "answer_correctness_score": correctness_score,
            "context_relevance_score": relevance_score,
            "citation_correctness_score": citation_score,
            "refusal_expected": ref_expected,
            "refusal_returned": ref_returned,
            "refusal_correct": ref_correct,
            "judge_status": judge_status,
            "judge_failure_reason": judge_reason,
            "latency": {
                "rag_pipeline_ms": pipeline_lat,
                "pipeline_ms": pipeline_lat,
                "judge_ms": judge_lat,
                "total_ms": total_eval_lat
            }
        })

    # Cleanup Isolated vector store & env
    store.clear()
    shutil.rmtree(temp_dir)
    os.environ.pop("UPLOAD_DIR")
    os.environ.pop("VECTOR_STORE_DIR")
    os.environ.pop("CHUNK_SIZE")
    os.environ.pop("CHUNK_OVERLAP")
    os.environ.pop("EVAL_QUALITY_TEMPERATURE")
    get_settings.cache_clear()

    # Aggregate results
    aggregates = aggregate_quality_metrics(per_question_results)

    # Return complete report
    return {
        "metadata": {
            "judge_provider": provider,
            "judge_model": model,
            "judge_temperature": temperature,
            "dataset_version": dataset["dataset_version"],
            "evaluation_version": "1.0",
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": per_question_results,
        "aggregates": aggregates["overall"],
        "categories": aggregates["categories"]
    }


def main():
    parser = argparse.ArgumentParser(description="NexusAI Quality and Faithfulness RAG Evaluator")
    parser.add_argument("--provider", type=str, choices=["mock", "openai", "gemini"], default=None, help="LLM Judge provider override")
    parser.add_argument("--model", type=str, default=None, help="LLM Judge model override")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM Judge temperature")
    parser.add_argument("--api-key", type=str, default=None, help="LLM Judge API key override")
    args = parser.parse_args()

    settings = get_settings()
    provider = args.provider or settings.EVAL_QUALITY_PROVIDER
    if args.model:
        model = args.model
    elif provider == "gemini" and settings.EVAL_QUALITY_PROVIDER != "gemini":
        model = "gemini-2.5-flash"
    else:
        model = settings.EVAL_QUALITY_MODEL

    print("==================================================")
    print(f"Starting NexusAI RAG Answer Quality Evaluation CLI")
    print(f"Judge Provider: {provider} | Model: {model}")
    print("==================================================")

    # 1. Load evaluation dataset
    dataset_path = Path(__file__).parent / "dataset.json"
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Run quality evaluation
    report = run_quality_evaluation(
        provider=provider,
        model=model,
        dataset=dataset,
        api_key=args.api_key,
        temperature=args.temperature
    )

    # 2. Persist Report
    report_dir = PROJECT_ROOT / "data" / "quality_benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"report_{int(time.time())}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # 3. Output ASCII summary
    print("\n==========================================================================================")
    print("                     EVALUATION QUALITY BENCHMARK REPORT SUMMARY                  ")
    print("==========================================================================================")
    print("Metric                 | Overall  | Direct   | MultiPart| CrossDoc | Unanswer | HallucCheck")
    print("------------------------------------------------------------------------------------------")
    
    overall = report["aggregates"]
    cats = report["categories"]

    def fmt(m_name, key, subkey):
        def _f(val):
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                return f"{val:.2f}"
            return str(val)

        val_overall = _f(overall.get(key, {}).get(subkey))
        val_direct = _f(cats.get("direct_answer", {}).get(key, {}).get(subkey))
        val_multi = _f(cats.get("multi_part", {}).get(key, {}).get(subkey))
        val_cross = _f(cats.get("cross_document", {}).get(key, {}).get(subkey))
        val_unans = _f(cats.get("unanswerable", {}).get(key, {}).get(subkey))
        val_halluc = _f(cats.get("hallucination_check", {}).get(key, {}).get(subkey))
        print(f"{m_name:<22} | {val_overall:<8} | {val_direct:<8} | {val_multi:<8} | {val_cross:<8} | {val_unans:<8} | {val_halluc:<8}")

    fmt("Hit@5", "retrieval", "hit_rate_at_5")
    fmt("MRR@5", "retrieval", "mrr_at_5")
    fmt("Correctness", "generation", "mean_correctness")
    fmt("Faithfulness", "generation", "mean_faithfulness")
    fmt("Context Relevance", "generation", "mean_context_relevance")
    fmt("Citation Correctness", "generation", "mean_citation_correctness")
    fmt("Refusal Accuracy", "safety", "refusal_accuracy")
    fmt("Latency (Mean ms)", "performance", "mean")
    print("==========================================================================================")
    print(f"Quality benchmark report successfully saved at: {report_file}")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
