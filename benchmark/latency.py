import numpy as np
from typing import List, Dict, Any, Union


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, Any]:
    """
    Compute rigorous latency percentiles (P50, P70, P90, P95, P99, P100, Mean).
    Also checks strict compliance against HH Goa <200ms requirement.
    """
    if not latencies:
        return {
            "p50": 0.0,
            "p70": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p100": 0.0,
            "mean": 0.0,
            "count": 0,
            "under_200ms": {
                "p50": True,
                "p70": True,
                "p90": True,
                "p95": True,
                "p99": True,
                "p100": True,
                "overall_compliant": True,
            },
        }

    arr = np.array(latencies, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p70 = float(np.percentile(arr, 70))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    p100 = float(np.max(arr))
    mean = float(np.mean(arr))

    under_200 = {
        "p50": bool(p50 <= 200.0),
        "p70": bool(p70 <= 200.0),
        "p90": bool(p90 <= 200.0),
        "p95": bool(p95 <= 200.0),
        "p99": bool(p99 <= 200.0),
        "p100": bool(p100 <= 200.0),
        "overall_compliant": bool(p100 <= 200.0),
    }

    return {
        "p50": round(p50, 3),
        "p70": round(p70, 3),
        "p90": round(p90, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "p100": round(p100, 3),
        "mean": round(mean, 3),
        "count": len(latencies),
        "under_200ms": under_200,
    }
