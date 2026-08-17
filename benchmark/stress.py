import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from generation.harness import RAGHarness
from benchmark.latency import compute_latency_percentiles


def run_stress_test(
    queries: List[str],
    concurrency_levels: List[int] = [10, 25, 50],
    strategy: str = "adaptive",
) -> Dict[str, Any]:
    """
    Execute high-concurrency stress test against the local pipeline (never calling real Sarvam).
    Measures Throughput (QPS), Latency Percentiles (P50–P100), and Error Rates.
    """
    harness = RAGHarness()
    results = {}

    for c in concurrency_levels:
        total_requests = len(queries)
        timings = []
        errors = 0
        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=c) as executor:
            future_to_query = {
                executor.submit(harness.process_rag_query, q, strategy, 5, True, True): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                try:
                    res = future.result()
                    timings.append(res.latency.total_ms)
                except Exception:
                    errors += 1

        total_wall_time_s = time.perf_counter() - t_start
        qps = total_requests / max(total_wall_time_s, 0.001)
        pcts = compute_latency_percentiles(timings)

        results[f"concurrency_{c}"] = {
            "concurrency": c,
            "total_requests": total_requests,
            "successful_requests": len(timings),
            "errors": errors,
            "error_rate_pct": round((errors / max(total_requests, 1)) * 100.0, 2),
            "total_duration_s": round(total_wall_time_s, 3),
            "throughput_qps": round(qps, 2),
            "latencies": pcts,
        }

    return results
