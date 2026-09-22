import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import statistics

from app.config import get_settings, Settings, PROJECT_ROOT
from app.services.vector_store import get_vector_store
from app.services.embedding_service import get_embedding_service
from app.services.chunking_service import chunk_document
from app.services.document_extractor import extract_document
from app.services.indexing_service import index_chunks
from app.services import metadata_service
from app.evaluation.evaluator import calculate_keyword_match_score, LLMJudgeEvaluator
from app.services.query_decomposition_service import search_with_decomposition

# Synthetic long texts generators to ensure they are ~2000 words each
def generate_hr_policy() -> str:
    content_list = []
    # Section 1
    content_list.append("SECTION 1: Office Hours and Work Schedules")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for office hours. The standard operational business hours for NexusAI offices are scheduled from 9:00 AM to 5:00 PM EST, Monday through Friday, excluding public holidays. All full-time employees are expected to be available during these core operational hours to facilitate collaboration, communication, and client meetings. To support personal schedules and promote productivity, flex-time schedule options allow employees to begin their workday anytime between 7:30 AM and 10:00 AM, adjusting their end time accordingly to fulfill a full eight-hour work shift daily. All working employees must sign and submit their physical or digital weekly attendance logs to their direct managers by Friday at 5:00 PM EST to ensure accurate payroll calculations and compliance records."
        )
    # Section 2
    content_list.append("SECTION 2: Leaves, PTO, and Sick Time")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for leaves. Full-time employees accrue 25 annual leave days per year, credited on January 1st of each calendar year. These days are designed to allow employees adequate rest, relaxation, and personal time off to recharge. Part-time employees accrue a prorated amount of annual leave based on their average weekly working hours. Prorated balances can be checked directly through the HR employee benefits platform at the end of every quarter. Paid sick leave is capped at 10 paid days per year for all employees. Sick leave is intended strictly for personal illness, medical appointments, or caring for immediate family members under medical guidance. Parental leave is 12 weeks fully paid for primary caregivers, which can be taken within the first year of a child's birth or adoption. Non-primary caregivers are eligible for 4 weeks of paid parental leave."
        )
    # Section 3
    content_list.append("SECTION 3: Remote Work & Hybrid Schedule")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for remote hybrid schedules. NexusAI supports hybrid work options. Employees can work remotely up to 2 days per week, with manager approval required. Hybrid arrangements require a stable home work environment to ensure productivity remains high. To offset internet and utility expenses incurred during hybrid work hours, employees receive a home office stipend of 50 dollars per month, processed alongside regular monthly expense reports. A minimum internet bandwidth of 50 Mbps download speed is required for remote hybrid work to ensure seamless participation in video conferences and access to corporate servers."
        )
    # Section 4
    content_list.append("SECTION 4: Dress Code and Conduct")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for dress code. The professional dress code at NexusAI is smart casual. This allows employees to dress comfortably while maintaining a professional appearance appropriate for meeting clients or hosting corporate guests. Professional conduct must be maintained on all corporate communication and messaging platforms, including Slack, email, and video calls. Respectful and inclusive interactions are mandatory at all times. Smart casual dress code includes collared shirts, blouses, knitwear, and clean dark denim."
        )
    return "\n\n".join(content_list)


def generate_benefits_policy() -> str:
    content_list = []
    # Section 1
    content_list.append("SECTION 1: Health & Dental Insurance Plans")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for health insurance coverage. NexusAI offers two health insurance plans (Standard HMO and Premium PPO). Standard HMO has a deductible of 500 dollars per individual. Premium PPO has a 200 dollar deductible. Medical checkups reward employees with standard care benefits. Coverage levels vary depending on individual choices and enrollments, which occur during open windows."
        )
    # Section 2
    content_list.append("SECTION 2: Wellness Program & Gym Benefits")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for wellness and gym. Employees can claim up to 75 dollars per month for gym membership reimbursement. Wellness checkups reward employees with a 100 dollar annual bonus. Staying healthy is a key priority for corporate wellness initiatives at NexusAI."
        )
    # Section 3
    content_list.append("SECTION 3: Travel, Expense, & Reimbursements")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for travel expenses. Business travel requires pre-authorization. Reimbursement requires receipts for any expense above 25 dollars. Expense reports must be submitted by the end of the month. Lodging reimbursement is capped at 200 dollars per night."
        )
    # Section 4
    content_list.append("SECTION 4: Retirement Savings (401k) Plan")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for retirement matching. NexusAI matches 100% of employee contributions up to 4% of base salary. The vesting period for matched contributions is 2 years. Vesting timelines are calculated from the start of employment."
        )
    return "\n\n".join(content_list)


def generate_security_policy() -> str:
    content_list = []
    # Section 1
    content_list.append("SECTION 1: Password and Authentication Rules")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for security passphrases. Passwords must be at least 12 characters, including uppercase, lowercase, numbers, and symbols. Passwords must be rotated every 90 days. Multi-Factor Authentication (MFA) is mandatory. Strong passphrases prevent unauthorized access and secure client databases."
        )
    # Section 2
    content_list.append("SECTION 2: Equipment and Software Usage")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for hardware usage. Only corporate-issued computers are allowed. Software installations must be pre-approved by IT. Personal browsing is restricted to break times."
        )
    # Section 3
    content_list.append("SECTION 3: Phishing and Incident Response")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for phishing incident workflows. Suspicious emails must be forwarded to security@nexusai.com. IT will verify and isolate systems within 15 minutes of any confirmed incident."
        )
    # Section 4
    content_list.append("SECTION 4: Data Classification and Privacy")
    for i in range(15):
        content_list.append(
            f"Paragraph {i+1} details for privacy data classifications. Data is classified as Public, Internal, or Confidential. Client personal data access is restricted to authorized engineers on a need-to-know basis."
        )
    return "\n\n".join(content_list)


# Synthetic Evaluation Documents Corpus
SYNTHETIC_CORPUS = {
    "11111111-1111-1111-1111-111111111111": {
        "filename": "company_policy.txt",
        "content": generate_hr_policy(),
    },
    "22222222-2222-2222-2222-222222222222": {
        "filename": "benefits_policy.txt",
        "content": generate_benefits_policy(),
    },
    "33333333-3333-3333-3333-333333333333": {
        "filename": "security_policy.txt",
        "content": generate_security_policy(),
    },
}


def calculate_hit_at_k(
    retrieved_chunks: List[Any], expected_docs: List[str], k: int
) -> Optional[int]:
    """Calculates Hit@K.

    Returns:
      - None if expected_docs is empty (unanswerable questions)
      - 1 if expected documents match criteria (all for cross-document, at least one for single-document)
      - 0 otherwise
    """
    if not expected_docs:
        return None

    retrieved_k = retrieved_chunks[:k]
    retrieved_doc_ids = {chunk.document_id for chunk in retrieved_k}

    # Cross-document success: all expected docs must be present
    # Single-document success: at least one matching expected doc present
    for expected_doc in expected_docs:
        if expected_doc not in retrieved_doc_ids:
            return 0
    return 1


def calculate_mrr_at_k(
    retrieved_chunks: List[Any], expected_docs: List[str], k: int
) -> Optional[float]:
    """Calculates MRR@K using the worst-performing rank for cross-document queries."""
    if not expected_docs:
        return None

    retrieved_k = retrieved_chunks[:k]
    ranks = []

    for expected_doc in expected_docs:
        first_rank = None
        for idx, chunk in enumerate(retrieved_k):
            if chunk.document_id == expected_doc:
                first_rank = idx + 1
                break
        if first_rank is None:
            # If any required document is missing, MRR is 0.0
            return 0.0
        ranks.append(first_rank)

    # Use the worst required document rank
    worst_rank = max(ranks)
    return float(1.0 / worst_rank)


def compute_aggregates(values: List[float]) -> Dict[str, Any]:
    """Computes mean, median, p95, and sample_count metrics."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "sample_count": 0,
        }

    # Use numpy for p95 calculation
    p95 = float(np.percentile(values, 95))
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": p95,
        "sample_count": len(values),
    }


def run_configuration_evaluation(
    chunk_size: int, chunk_overlap: int, dataset: Dict[str, Any]
) -> Dict[str, Any]:
    """Sets up an isolated FAISS instance, indexes the synthetic corpus, and runs evaluation."""
    # Create isolated temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_raw = Path(temp_dir) / "raw"
    temp_vectors = Path(temp_dir) / "vectors"
    temp_raw.mkdir(parents=True)
    temp_vectors.mkdir(parents=True)

    # Temporarily override configuration using global environment simulation
    os.environ["UPLOAD_DIR"] = str(temp_raw)
    os.environ["VECTOR_STORE_DIR"] = str(temp_vectors)
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)

    get_settings.cache_clear()
    settings = get_settings()

    # Reset vector store to ensure isolation
    store = get_vector_store()
    store.clear()

    chunks_per_doc = {}
    total_chunks = 0

    # 1. Index Synthetic Corpus via production pipelines
    for doc_id, doc_info in SYNTHETIC_CORPUS.items():
        # Save raw file
        raw_file_path = temp_raw / doc_info["filename"]
        with open(raw_file_path, "w", encoding="utf-8") as f:
            f.write(doc_info["content"])

        # Save metadata sidecar using metadata_service
        metadata_service.create_metadata(
            document_id=doc_id,
            original_filename=doc_info["filename"],
            stored_filename=doc_info["filename"],
            file_type="txt",
            file_size=len(doc_info["content"]),
        )

        # Extract, chunk, and index using production services
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

        # Track chunk layout variation
        chunks_per_doc[doc_id] = len(chunks)
        total_chunks += len(chunks)

    # 2. Run Test Questions
    from app.services.llm_service import get_llm_service
    llm_service = get_llm_service()
    evaluator = LLMJudgeEvaluator(
        provider=settings.EVALUATOR_PROVIDER, model=settings.EVALUATOR_MODEL
    )

    per_question_results = []
    latencies = {
        "embedding": [],
        "retrieval": [],
        "llm": [],
        "total": [],
    }
    keyword_scores = []

    hit_scores = []
    mrr_scores = []

    for q in dataset["questions"]:
        question_id = q["id"]
        question_text = q["question"]
        expected_ans = q.get("expected_answer")
        expected_docs = q.get("expected_source_documents", [])
        category = q["category"]

        # Perform Search and LLM QA manually timing each step
        embedding_service = get_embedding_service()

        # Embedding & Retrieval Latency
        start_retrieve = time.perf_counter()
        search_results = search_with_decomposition(store, question_text, top_k=5)
        retrieve_time = (time.perf_counter() - start_retrieve) * 1000.0
        embed_time = 0.0
        latencies["retrieval"].append(retrieve_time)
        latencies["embedding"].append(embed_time)

        # Calculate retrieval metrics
        # search_results is list of (Chunk, score)
        chunks_only = [item[0] for item in search_results]
        retrieved_source_docs = list(
            dict.fromkeys(chunk.document_id for chunk in chunks_only)
        )

        hit = calculate_hit_at_k(chunks_only, expected_docs, k=5)
        mrr = calculate_mrr_at_k(chunks_only, expected_docs, k=5)

        if hit is not None:
            hit_scores.append(hit)
        if mrr is not None:
            mrr_scores.append(mrr)

        first_relevant_rank = None
        if expected_docs:
            ranks = []
            for doc in expected_docs:
                for idx, chunk in enumerate(chunks_only):
                    if chunk.document_id == doc:
                        ranks.append(idx + 1)
                        break
            if len(ranks) == len(expected_docs):
                first_relevant_rank = max(ranks)

        # LLM Generation Latency
        highest_score = max(score for _, score in search_results) if search_results else 0.0

        fallback_expected = True if category == "unanswerable" else False
        fallback_returned = None
        ans_text = ""
        llm_time = 0.0

        if highest_score < settings.LLM_RELEVANCE_THRESHOLD:
            # Fallback direct response
            ans_text = "I could not find information about this in the provided documents."
            if category == "unanswerable":
                fallback_returned = True
        else:
            if category == "unanswerable":
                fallback_returned = False

            # Build context and run LLM
            from app.services.context_builder import build_context
            # Deduplicate
            seen_chunks = {}
            for chunk, score in search_results:
                if score < settings.LLM_RELEVANCE_THRESHOLD:
                    continue
                cid = chunk.chunk_id
                if cid not in seen_chunks or score > seen_chunks[cid][1]:
                    seen_chunks[cid] = (chunk, score)

            filtered_chunks = [item[0] for item in seen_chunks.values()]
            context_text = build_context(filtered_chunks)

            start_llm = time.perf_counter()
            ans_text = llm_service.generate_answer(question_text, context_text)
            llm_time = (time.perf_counter() - start_llm) * 1000.0
            latencies["llm"].append(llm_time)

            if category == "hallucination_check":
                # Check for refusal when query matches absent terms
                if "go" in question_text.lower() and "go" not in context_text.lower():
                    if ans_text == "I could not find information about this in the provided documents.":
                        fallback_returned = True
                    else:
                        fallback_returned = False

        total_time = embed_time + retrieve_time + llm_time
        latencies["total"].append(total_time)

        # Keyword overlap match score
        kw_score = 0.0
        if expected_ans and q.get("keywords"):
            kw_score = calculate_keyword_match_score(ans_text, q["keywords"])
            keyword_scores.append(kw_score)

        # LLM Judge evaluation (default mock returns 1.0 success)
        judge_score, eval_status, _ = evaluator.evaluate_correctness(
            question_text, ans_text, expected_ans
        )

        per_question_results.append(
            {
                "question_id": question_id,
                "category": category,
                "retrieved_source_documents": retrieved_source_docs,
                "expected_source_documents": expected_docs,
                "first_relevant_rank": first_relevant_rank,
                "hit_at_k": hit,
                "reciprocal_rank": mrr,
                "fallback_expected": fallback_expected,
                "fallback_returned": fallback_returned,
                "top_score": float(highest_score),
                "latency": {
                    "embedding_ms": embed_time,
                    "retrieval_ms": retrieve_time,
                    "llm_ms": llm_time,
                    "total_ms": total_time,
                },
                "keyword_match_score": kw_score,
                "evaluation_status": eval_status,
            }
        )

    # Calculate aggregate scores
    avg_hit = float(statistics.mean(hit_scores)) if hit_scores else 0.0
    avg_mrr = float(statistics.mean(mrr_scores)) if mrr_scores else 0.0
    avg_kw = float(statistics.mean(keyword_scores)) if keyword_scores else 0.0

    agg_embedding = compute_aggregates(latencies["embedding"])
    agg_retrieval = compute_aggregates(latencies["retrieval"])
    agg_llm = compute_aggregates(latencies["llm"])
    agg_total = compute_aggregates(latencies["total"])

    # Clean up isolated FAISS store index files and raw folder completely
    store.clear()
    shutil.rmtree(temp_dir)

    # Reset environment variables
    os.environ.pop("UPLOAD_DIR")
    os.environ.pop("VECTOR_STORE_DIR")
    os.environ.pop("CHUNK_SIZE")
    os.environ.pop("CHUNK_OVERLAP")
    get_settings.cache_clear()

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "results": per_question_results,
        "aggregates": {
            "chunks_per_doc": chunks_per_doc,
            "total_chunks": total_chunks,
            "retrieval": {
                "hit_rate_at_5": avg_hit,
                "mrr_at_5": avg_mrr,
            },
            "keyword_match_score": avg_kw,
            "latency": {
                "embedding": agg_embedding,
                "retrieval": agg_retrieval,
                "llm": agg_llm,
                "total": agg_total,
            },
        },
    }


def check_is_discriminative(runs: List[Dict[str, Any]]) -> bool:
    """Verifies that chunk configurations actually produce different chunk counts."""
    if len(runs) < 2:
        return True
    
    total_chunk_counts = [run["aggregates"]["total_chunks"] for run in runs]
    # Check if all counts are identical
    return len(set(total_chunk_counts)) > 1


def main():
    print("==================================================")
    print("Starting NexusAI RAG Evaluation & Benchmarking CLI")
    print("==================================================")

    # 1. Load evaluation dataset
    dataset_path = Path(__file__).parent / "dataset.json"
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded evaluation dataset version: {dataset['dataset_version']}")
    print(f"Total test questions: {len(dataset['questions'])}")

    # Group questions by category for logging
    categories = {}
    for q in dataset["questions"]:
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("Category distribution:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count}")

    # Print corpus details
    print("Synthetic Evaluation Corpus loaded:")
    for doc_id, doc_info in SYNTHETIC_CORPUS.items():
        word_count = len(doc_info["content"].split())
        print(f"  - Document {doc_id} ({doc_info['filename']}): {word_count} words")

    # 2. Configurations to test
    configs = [
        {"chunk_size": 500, "chunk_overlap": 50},
        {"chunk_size": 800, "chunk_overlap": 80},
        {"chunk_size": 1000, "chunk_overlap": 100},
    ]

    benchmark_runs = []

    # Save original env variables to restore later
    orig_env = {
        "UPLOAD_DIR": os.environ.get("UPLOAD_DIR"),
        "VECTOR_STORE_DIR": os.environ.get("VECTOR_STORE_DIR"),
        "CHUNK_SIZE": os.environ.get("CHUNK_SIZE"),
        "CHUNK_OVERLAP": os.environ.get("CHUNK_OVERLAP"),
    }

    try:
        for idx, config in enumerate(configs):
            size = config["chunk_size"]
            overlap = config["chunk_overlap"]
            print(
                f"\nExecuting Configuration {idx + 1}/3: chunk_size={size}, overlap={overlap}..."
            )
            run_data = run_configuration_evaluation(size, overlap, dataset)
            benchmark_runs.append(run_data)
            
            # Print chunks stats per doc
            print(f"Finished Configuration {idx + 1}. Chunk layout details:")
            for doc_id, count in run_data["aggregates"]["chunks_per_doc"].items():
                print(f"  - Doc {doc_id}: {count} chunks")
            print(f"  - Total chunks: {run_data['aggregates']['total_chunks']}")
    finally:
        # Restore environment settings to prevent side-effects on production defaults
        for k, v in orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        get_settings.cache_clear()

    # 3. Perform Benchmark Discriminative Check
    is_discriminative = check_is_discriminative(benchmark_runs)
    if not is_discriminative:
        print("\n[WARNING] Benchmark is non-discriminative! All tested configurations produced identical chunk counts.")
    else:
        print("\nBenchmark is discriminative. Chunk counts differ across tested configurations.")

    # 4. Create Benchmark Report
    settings = get_settings()
    report = {
        "benchmark_id": f"bench-{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_discriminative": is_discriminative,
        "reproducibility": {
            "dataset_version": dataset["dataset_version"],
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "top_k": 5,
            "relevance_threshold": settings.LLM_RELEVANCE_THRESHOLD,
        },
        "configurations": benchmark_runs,
    }

    # Persist report
    benchmarks_dir = PROJECT_ROOT / "data" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    report_file = benchmarks_dir / f"benchmark_{int(time.time())}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # 5. Print Summary ASCII Table
    print("\n==========================================================================")
    print("                     EVALUATION BENCHMARK SUMMARY REPORT                  ")
    print("==========================================================================")
    print("Config         | Hit@5  | MRR@5  | KeyMatch | Mean Latency | p95 Latency | Chunks")
    print("--------------------------------------------------------------------------")
    for run in benchmark_runs:
        size = run["chunk_size"]
        overlap = run["chunk_overlap"]
        hit = run["aggregates"]["retrieval"]["hit_rate_at_5"]
        mrr = run["aggregates"]["retrieval"]["mrr_at_5"]
        kw = run["aggregates"]["keyword_match_score"]
        mean_lat = run["aggregates"]["latency"]["total"]["mean"]
        p95_lat = run["aggregates"]["latency"]["total"]["p95"]
        tot_chunks = run["aggregates"]["total_chunks"]

        config_str = f"cs={size},ov={overlap}"
        print(
            f"{config_str:<14} | {hit:<6.2f} | {mrr:<6.2f} | {kw:<8.2f} | {mean_lat:<12.2f} | {p95_lat:<11.2f} | {tot_chunks:<6}"
        )
    print("==========================================================================")
    print(f"Benchmark results persisted at: {report_file}")
    print("Indicative statement: p95 latency results on a small benchmark dataset are")
    print("indicative rather than a statistically strong performance guarantee.")
    print("==========================================================================")


if __name__ == "__main__":
    main()
