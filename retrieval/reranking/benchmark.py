import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from retrieval.dense.retriever import DenseRetriever
from retrieval.lexical.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranking.model import CrossEncoderReranker, CustomReranker
from retrieval.reranking.reranker import RerankerService
from retrieval.pipeline import RetrievalPipeline
from retrieval.evaluation.evaluator import RetrievalEvaluator
from retrieval.evaluation.failures import FailureAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.reranking.benchmark")


def benchmark_multilingual_smoke_test(reranker: CrossEncoderReranker) -> List[Dict[str, Any]]:
    """
    Multilingual smoke test verifying cross-encoder inference across 7 languages.
    """
    logger.info("Executing Multilingual Cross-Encoder Smoke Test across 7 languages...")
    smoke_data = [
        {
            "language": "Hindi (hi)",
            "query": "भारत की राजधानी क्या है?",
            "relevant_doc": "नई दिल्ली भारत की आधिकारिक राजधानी और सरकार का केंद्र है।",
            "irrelevant_doc": "सेब एक बहुत ही स्वादिष्ट और स्वास्थ्यवर्धक फल है।",
        },
        {
            "language": "English (en)",
            "query": "What is the capital of India?",
            "relevant_doc": "New Delhi is the official capital of India and the seat of government.",
            "irrelevant_doc": "Apples are nutritious and healthy fruits grown in temperate climates.",
        },
        {
            "language": "Hinglish (hi-Latn)",
            "query": "India ki capital kya hai?",
            "relevant_doc": "New Delhi India ki rajdhani hai aur government ka center hai.",
            "irrelevant_doc": "Apple ek healthy fruit hota hai jo thand mein ugta hai.",
        },
        {
            "language": "Bengali (bn)",
            "query": "ভারতের রাজধানী কী?",
            "relevant_doc": "নতুন দিল্লি হলো ভারতের সরকারি রাজধানী ও প্রশাসনিক কেন্দ্র।",
            "irrelevant_doc": "আপেল একটি সুস্বাদু এবং স্বাস্থ্যকর পুষ্টিকর ফল।",
        },
        {
            "language": "Tamil (ta)",
            "query": "இந்தியாவின் தலைநகரம் எது?",
            "relevant_doc": "புது தில்லி இந்தியாவின் அதிகாரப்பூர்வ தலைநகரம் மற்றும் அரசு மையம் ஆகும்.",
            "irrelevant_doc": "ஆப்பிள் ஒரு சுவையான மற்றும் ஆரோக்கியமான பழமாகும்.",
        },
        {
            "language": "Telugu (te)",
            "query": "భారతదేశ రాజధాని ఏది?",
            "relevant_doc": "న్యూఢిల్లీ భారతదేశ అధికారిక రాజధాని మరియు ప్రభుత్వ కేంద్రం.",
            "irrelevant_doc": "యాపిల్ చాలా రుచికరమైన మరియు ఆరోగ్యకరమైన పండు.",
        },
        {
            "language": "Marathi (mr)",
            "query": "भारताची राजधानी कोणती आहे?",
            "relevant_doc": "नवी दिल्ली ही भारताची अधिकृत राजधानी आणि प्रशासकीय केंद्र आहे.",
            "irrelevant_doc": "सफरचंद हे एक अतिशय चवदार आणि पौष्टिक फळ आहे.",
        },
    ]

    results = []
    for item in smoke_data:
        t0 = time.perf_counter()
        scores = reranker.score(item["query"], [item["relevant_doc"], item["irrelevant_doc"]])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rel_score = scores[0]
        irrel_score = scores[1]
        passed_ordering = rel_score > irrel_score

        res = {
            "language": item["language"],
            "query": item["query"],
            "relevant_score": rel_score,
            "irrelevant_score": irrel_score,
            "score_differential": round(rel_score - irrel_score, 4),
            "correct_ranking": passed_ordering,
            "inference_ms": round(elapsed_ms, 3),
            "device": reranker.device,
        }
        results.append(res)
        logger.info(f"[{item['language']}] Rel={rel_score:.4f} Irrel={irrel_score:.4f} (Ranked Correct: {passed_ordering}) in {elapsed_ms:.2f}ms")

    return results


def benchmark_batch_sizes(reranker: CrossEncoderReranker, query: str, passages: List[str]) -> Dict[int, float]:
    """
    Measure inference latency across different batch sizes (4, 8, 16, 32).
    """
    logger.info(f"Measuring batch size ablation on {len(passages)} candidate passages...")
    batch_latencies = {}
    orig_batch = reranker.batch_size

    for bs in [4, 8, 16, 32]:
        reranker.batch_size = bs
        # Run 2 warmup runs then 3 measured runs
        for _ in range(2):
            _ = reranker.score(query, passages)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = reranker.score(query, passages)
            times.append((time.perf_counter() - t0) * 1000.0)
        batch_latencies[bs] = round(float(np.mean(times)), 3)
        logger.info(f"Batch Size {bs}: {batch_latencies[bs]} ms for {len(passages)} passages")

    reranker.batch_size = orig_batch
    return batch_latencies


def run_comprehensive_reranker_benchmark(
    dataset_parquet_path: Path,
    strategies: List[str] = ["fixed", "sentence", "adaptive"],
    num_queries: int = 100,
) -> Dict[str, Any]:
    """
    Executes full multi-strategy retrieval evaluation with both Real Cross-Encoder and Custom Reranker,
    alongside cold/warm latency percentiles and multilingual validation.
    """
    logger.info(f"Loading {num_queries} queries from {dataset_parquet_path}...")
    df = pd.read_parquet(dataset_parquet_path)
    sample_df = df.head(num_queries)

    # 1. Cold vs Warm Initialization Measurement
    logger.info("\n--- BENCHMARKING COLD VS WARM INITIALIZATION ---")
    cold_t0 = time.perf_counter()
    real_reranker = CrossEncoderReranker(
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        use_cache=False,  # Measure raw model inference for benchmarks
        lazy_load=False,
    )
    # First inference included in cold time
    _ = real_reranker.score("भारत की राजधानी", ["नई दिल्ली"])
    cold_init_ms = (time.perf_counter() - cold_t0) * 1000.0

    warm_t0 = time.perf_counter()
    real_reranker.warmup()
    warm_init_ms = (time.perf_counter() - warm_t0) * 1000.0

    logger.info(f"Cold Initialization + First Inference: {cold_init_ms:.2f} ms")
    logger.info(f"Warm Inference: {warm_init_ms:.2f} ms")

    # 2. Multilingual Smoke Test
    multilingual_smoke = benchmark_multilingual_smoke_test(real_reranker)

    # 3. Batch Size Ablation
    sample_passages = [
        f"यह भारत के विभिन्न शहरों और इतिहास से संबंधित महत्वपूर्ण अनुच्छेद संख्या {i} है।" for i in range(20)
    ]
    batch_ablation = benchmark_batch_sizes(real_reranker, "भारत का इतिहास", sample_passages)

    # 4. Extract Query Ground Truths
    evaluator = RetrievalEvaluator()
    query_ground_truths: List[Dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        q_id = int(row["query_id"])
        q_text = row["query"]
        passages = row.get("passages", [])
        pos_ids: Set[str] = set()
        pos_text = ""

        if isinstance(passages, (list, np.ndarray)):
            for p in passages:
                if isinstance(p, dict):
                    if p.get("is_selected") == 1 or p.get("is_selected") is True:
                        pos_ids.add(str(p.get("passage_id", "")))
                        if not pos_text:
                            pos_text = str(p.get("text", ""))
        elif isinstance(passages, dict):
            p_ids = passages.get("passage_id", [])
            is_sels = passages.get("is_selected", [])
            p_texts = passages.get("passage_text", []) or passages.get("text", [])
            for p_id, is_sel, p_txt in zip(p_ids, is_sels, p_texts):
                if is_sel == 1 or is_sel is True:
                    pos_ids.add(str(p_id))
                    if not pos_text:
                        pos_text = str(p_txt)

        query_ground_truths.append({
            "query_id": q_id,
            "query": q_text,
            "ground_truth_passage_ids": pos_ids,
            "ground_truth_text": pos_text,
        })

    comparison_matrix: List[Dict[str, Any]] = []
    old_vs_new_comparison: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []
    real_reranker_latencies: Dict[str, Dict[str, Any]] = {}
    custom_reranker_latencies: Dict[str, Dict[str, Any]] = {}

    custom_reranker = CustomReranker(use_cache=False)
    custom_service = RerankerService(reranker=custom_reranker)
    real_service = RerankerService(reranker=real_reranker)

    for strat in strategies:
        logger.info(f"\n=======================================================")
        logger.info(f"  EVALUATING RETRIEVAL & RERANKING: {strat.upper()}")
        logger.info(f"=======================================================")

        dense_retriever = DenseRetriever(strategy=strat)
        bm25_retriever = BM25Retriever(strategy=strat)
        hybrid_retriever = HybridRetriever(
            strategy=strat,
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
        )

        # ABLATION A: Dense Only
        dense_metrics = []
        dense_lats = []
        for item in query_ground_truths:
            t0 = time.perf_counter()
            cands = dense_retriever.search(query=item["query"], strategy=strat, top_k=20)
            dense_lats.append((time.perf_counter() - t0) * 1000.0)
            m = evaluator.evaluate_query(cands, item["ground_truth_passage_ids"], item["query_id"])
            dense_metrics.append(m)

        dense_agg = evaluator.aggregate_metrics(dense_metrics)
        dense_agg["latency_ms"] = round(float(np.mean(dense_lats)), 3)
        comparison_matrix.append({
            "strategy": strat,
            "configuration": "Dense Only",
            "recall@1": dense_agg["recall@1"],
            "recall@3": dense_agg["recall@3"],
            "recall@5": dense_agg["recall@5"],
            "recall@10": dense_agg["recall@10"],
            "recall@20": dense_agg["recall@20"],
            "mrr": dense_agg["mrr"],
            "mean_latency_ms": dense_agg["latency_ms"],
        })

        # ABLATION B: BM25 Only
        bm25_metrics = []
        bm25_lats = []
        for item in query_ground_truths:
            t0 = time.perf_counter()
            cands = bm25_retriever.search(query=item["query"], top_k=20)
            bm25_lats.append((time.perf_counter() - t0) * 1000.0)
            m = evaluator.evaluate_query(cands, item["ground_truth_passage_ids"], item["query_id"])
            bm25_metrics.append(m)

        bm25_agg = evaluator.aggregate_metrics(bm25_metrics)
        bm25_agg["latency_ms"] = round(float(np.mean(bm25_lats)), 3)
        comparison_matrix.append({
            "strategy": strat,
            "configuration": "BM25 Only",
            "recall@1": bm25_agg["recall@1"],
            "recall@3": bm25_agg["recall@3"],
            "recall@5": bm25_agg["recall@5"],
            "recall@10": bm25_agg["recall@10"],
            "recall@20": bm25_agg["recall@20"],
            "mrr": bm25_agg["mrr"],
            "mean_latency_ms": bm25_agg["latency_ms"],
        })

        # ABLATION C: Hybrid (Dense + BM25)
        hybrid_metrics = []
        hybrid_lats = []
        fused_pools = []
        for item in query_ground_truths:
            t0 = time.perf_counter()
            out = hybrid_retriever.search(query=item["query"], strategy=strat, dense_k=20, bm25_k=20, final_k=20)
            hybrid_lats.append((time.perf_counter() - t0) * 1000.0)
            m = evaluator.evaluate_query(out["fused_candidates"], item["ground_truth_passage_ids"], item["query_id"])
            hybrid_metrics.append(m)
            fused_pools.append(out)

        hybrid_agg = evaluator.aggregate_metrics(hybrid_metrics)
        hybrid_agg["latency_ms"] = round(float(np.mean(hybrid_lats)), 3)
        comparison_matrix.append({
            "strategy": strat,
            "configuration": "Hybrid (Dense+BM25)",
            "recall@1": hybrid_agg["recall@1"],
            "recall@3": hybrid_agg["recall@3"],
            "recall@5": hybrid_agg["recall@5"],
            "recall@10": hybrid_agg["recall@10"],
            "recall@20": hybrid_agg["recall@20"],
            "mrr": hybrid_agg["mrr"],
            "mean_latency_ms": hybrid_agg["latency_ms"],
        })

        # ABLATION D1: Hybrid + CustomReranker (Old Handcrafted)
        custom_metrics = []
        custom_lats = []
        for idx, item in enumerate(query_ground_truths):
            fused = fused_pools[idx]["fused_candidates"]
            t0 = time.perf_counter()
            rerank_out = custom_service.rerank_candidates(query=item["query"], candidates=fused, top_k=8)
            custom_lats.append((time.perf_counter() - t0) * 1000.0)
            m = evaluator.evaluate_query(rerank_out["reranked_candidates"], item["ground_truth_passage_ids"], item["query_id"], ks=[1, 3, 5, 8])
            custom_metrics.append(m)

        custom_agg = evaluator.aggregate_metrics(custom_metrics)
        custom_agg["latency_ms"] = round(float(np.mean(custom_lats)), 3)
        custom_reranker_latencies[strat] = {
            "p50_ms": round(float(np.percentile(custom_lats, 50)), 3),
            "p70_ms": round(float(np.percentile(custom_lats, 70)), 3),
            "p90_ms": round(float(np.percentile(custom_lats, 90)), 3),
            "p95_ms": round(float(np.percentile(custom_lats, 95)), 3),
            "p99_ms": round(float(np.percentile(custom_lats, 99)), 3),
            "p100_ms": round(float(np.max(custom_lats)), 3),
            "mean_ms": round(float(np.mean(custom_lats)), 3),
        }

        # ABLATION D2: Hybrid + Real Cross-Encoder (New Transformer)
        real_metrics = []
        real_lats = []
        pipeline_real = RetrievalPipeline(
            strategy=strat,
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            reranker_service=real_service,
        )

        for idx, item in enumerate(query_ground_truths):
            t0 = time.perf_counter()
            out = pipeline_real.retrieve(
                query=item["query"],
                strategy=strat,
                dense_k=20,
                bm25_k=20,
                hybrid_k=20,
                rerank_top_k=8,
                enable_reranking=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            real_lats.append(elapsed_ms)
            m = evaluator.evaluate_query(out["reranked_results"], item["ground_truth_passage_ids"], item["query_id"], ks=[1, 3, 5, 8])
            real_metrics.append(m)

            # Failure Analysis on misses (recall@5 == 0)
            if m.get("recall@5", 0.0) == 0.0:
                failure_rec = FailureAnalyzer.record_failure(
                    query_id=item["query_id"],
                    query=item["query"],
                    expected_passage_ids=item["ground_truth_passage_ids"],
                    pipeline_output=out,
                    ground_truth_text=item["ground_truth_text"],
                )
                if len(all_failures) < 25:
                    all_failures.append(failure_rec)

        real_agg = evaluator.aggregate_metrics(real_metrics)
        real_agg["latency_ms"] = round(float(np.mean(real_lats)), 3)
        comparison_matrix.append({
            "strategy": strat,
            "configuration": "Hybrid + Real Cross-Encoder",
            "recall@1": real_agg["recall@1"],
            "recall@3": real_agg["recall@3"],
            "recall@5": real_agg["recall@5"],
            "recall@10": real_agg.get("recall@10", real_agg["recall@5"]),
            "recall@20": real_agg.get("recall@20", real_agg["recall@5"]),
            "mrr": real_agg["mrr"],
            "mean_latency_ms": real_agg["latency_ms"],
        })

        real_reranker_latencies[strat] = {
            "p50_ms": round(float(np.percentile(real_lats, 50)), 3),
            "p70_ms": round(float(np.percentile(real_lats, 70)), 3),
            "p90_ms": round(float(np.percentile(real_lats, 90)), 3),
            "p95_ms": round(float(np.percentile(real_lats, 95)), 3),
            "p99_ms": round(float(np.percentile(real_lats, 99)), 3),
            "p100_ms": round(float(np.max(real_lats)), 3),
            "mean_ms": round(float(np.mean(real_lats)), 3),
        }

        # Head-to-Head Old vs New Comparison Entry
        old_vs_new_comparison.append({
            "strategy": strat,
            "custom_reranker": {
                "recall@1": custom_agg["recall@1"],
                "recall@5": custom_agg["recall@5"],
                "mrr": custom_agg["mrr"],
                "mean_latency_ms": custom_agg["latency_ms"],
                "p50_ms": custom_reranker_latencies[strat]["p50_ms"],
                "p95_ms": custom_reranker_latencies[strat]["p95_ms"],
                "device": "cpu",
                "model_type": "Handcrafted Lexical+Semantic Projections",
            },
            "cross_encoder_reranker": {
                "recall@1": real_agg["recall@1"],
                "recall@5": real_agg["recall@5"],
                "mrr": real_agg["mrr"],
                "mean_latency_ms": real_agg["latency_ms"],
                "p50_ms": real_reranker_latencies[strat]["p50_ms"],
                "p95_ms": real_reranker_latencies[strat]["p95_ms"],
                "device": real_reranker.device,
                "model_type": "Pretrained Transformer Cross-Encoder (117M params)",
                "model_name": real_reranker.model_name,
            },
        })

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_info": real_reranker.get_model_info(),
        "initialization": {
            "cold_ms": round(cold_init_ms, 3),
            "warm_ms": round(warm_init_ms, 3),
        },
        "multilingual_smoke_test": multilingual_smoke,
        "batch_size_ablation": batch_ablation,
        "comparison_matrix": comparison_matrix,
        "old_vs_new_comparison": old_vs_new_comparison,
        "latency_percentiles_real_reranker": real_reranker_latencies,
        "latency_percentiles_custom_reranker": custom_reranker_latencies,
        "failures": all_failures,
    }


def save_reports(eval_data: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = output_dir / "reranker_comparison.json"
    comparison_md = output_dir / "reranker_comparison.md"
    latency_json = output_dir / "reranker_latency.json"
    failures_json = output_dir / "reranker_failures.json"

    # Save JSON files
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    latency_data = {
        "model": eval_data["model_info"],
        "initialization": eval_data["initialization"],
        "batch_size_ablation": eval_data["batch_size_ablation"],
        "real_cross_encoder_latencies": eval_data["latency_percentiles_real_reranker"],
        "custom_reranker_latencies": eval_data["latency_percentiles_custom_reranker"],
    }
    with open(latency_json, "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2, ensure_ascii=False)

    FailureAnalyzer.save_failures(eval_data["failures"], failures_json)

    # Generate Markdown Table
    ablation_rows = []
    for r in eval_data["comparison_matrix"]:
        ablation_rows.append(
            f"| **{r['strategy'].capitalize()}** | {r['configuration']} | "
            f"{r['recall@1']:.3f} | {r['recall@5']:.3f} | {r.get('recall@10', 0.0):.3f} | "
            f"{r['mrr']:.3f} | **{r['mean_latency_ms']:.2f} ms** |"
        )
    ablation_str = "\n".join(ablation_rows)

    lat_rows = []
    for s, l in eval_data["latency_percentiles_real_reranker"].items():
        lat_rows.append(
            f"| **{s.capitalize()}** | {l['p50_ms']} ms | {l['p70_ms']} ms | {l['p90_ms']} ms | "
            f"{l['p95_ms']} ms | {l['p99_ms']} ms | {l['p100_ms']} ms | {l['mean_ms']} ms |"
        )
    lat_str = "\n".join(lat_rows)

    smoke_rows = []
    for s in eval_data["multilingual_smoke_test"]:
        status = "PASSED" if s["correct_ranking"] else "FAILED"
        smoke_rows.append(
            f"| **{s['language']}** | `{s['query']}` | {s['relevant_score']:.4f} | {s['irrelevant_score']:.4f} | "
            f"+{s['score_differential']:.4f} | **{status}** | {s['inference_ms']} ms |"
        )
    smoke_str = "\n".join(smoke_rows)

    old_new_rows = []
    for comp in eval_data["old_vs_new_comparison"]:
        c = comp["custom_reranker"]
        r = comp["cross_encoder_reranker"]
        old_new_rows.append(
            f"| **{comp['strategy'].capitalize()}** | CustomReranker (Handcrafted) | {c['recall@1']:.3f} | {c['recall@5']:.3f} | {c['mrr']:.3f} | {c['mean_latency_ms']:.2f} ms | {c['p95_ms']} ms |\n"
            f"| **{comp['strategy'].capitalize()}** | **CrossEncoderReranker (Transformer)** | **{r['recall@1']:.3f}** | **{r['recall@5']:.3f}** | **{r['mrr']:.3f}** | **{r['mean_latency_ms']:.2f} ms** | **{r['p95_ms']} ms** |"
        )
    old_new_str = "\n".join(old_new_rows)

    md_content = f"""# Module 5.1 — Real Multilingual Cross-Encoder Reranker Report

**HH Goa 2026 — Task 2 | Module 5.1: Real Cross-Encoder Implementation & Evaluation**  
*Evaluation Model: `{eval_data['model_info']['model']}` on {eval_data['model_info']['device'].upper()}*

---

## 1. Multilingual Verification Smoke Test (7 Languages)

| Language | Test Query | Relevant Score | Irrelevant Score | Delta | Verification Status | Latency |
|---|---|---|---|---|---|---|
{smoke_str}

---

## 2. Retrieval Ablation Matrix (100 MSMARCO-XI Hindi Validation Queries)

| Strategy | Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Mean Warm Latency |
|---|---|---|---|---|---|---|
{ablation_str}

---

## 3. Old Custom Reranker vs New Real Cross-Encoder Comparison

| Strategy | Reranker Architecture | Recall@1 | Recall@5 | MRR | Mean Latency | P95 Latency |
|---|---|---|---|---|---|---|
{old_new_str}

---

## 4. Warm Latency Percentiles (End-to-End Hybrid + Real Cross-Encoder)

| Strategy | P50 | P70 | P90 | P95 | P99 | P100 (Max) | Mean |
|---|---|---|---|---|---|---|---|
{lat_str}

---

## 5. Batch Size Ablation (20 Candidates on CPU)

| Batch Size | Inference Latency |
|---|---|
| **4** | {eval_data['batch_size_ablation'].get(4, 'N/A')} ms |
| **8 (Default)** | {eval_data['batch_size_ablation'].get(8, 'N/A')} ms |
| **16** | {eval_data['batch_size_ablation'].get(16, 'N/A')} ms |
| **32** | {eval_data['batch_size_ablation'].get(32, 'N/A')} ms |

---

## 6. Cold vs Warm Initialization

- **Cold Initialization (Model Load + First Forward Pass)**: `{eval_data['initialization']['cold_ms']} ms`
- **Warm Inference**: `{eval_data['initialization']['warm_ms']} ms`
"""

    with open(comparison_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved reranker comparison markdown -> {comparison_md}")
    logger.info(f"Saved reranker comparison JSON -> {comparison_json}")
    logger.info(f"Saved reranker latency JSON -> {latency_json}")
    logger.info(f"Saved failures JSON -> {failures_json}")


def main():
    dataset_path = BASE_DIR / "data" / "processed" / "msmarco_xi_hi_validation.parquet"
    out_dir = BASE_DIR / "data" / "statistics"

    eval_data = run_comprehensive_reranker_benchmark(
        dataset_parquet_path=dataset_path,
        strategies=["fixed", "sentence", "adaptive"],
        num_queries=100,
    )
    save_reports(eval_data, out_dir)


if __name__ == "__main__":
    main()
