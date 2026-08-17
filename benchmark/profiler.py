import time
import os
from typing import Dict, Any, Callable, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class SystemProfiler:
    """
    Tracks execution times, CPU utilization, and RAM consumption.
    Measures cold-start vs warm-request transitions.
    """

    def __init__(self):
        if _HAS_PSUTIL:
            try:
                self.process = psutil.Process(os.getpid())
            except Exception:
                self.process = None
        else:
            self.process = None

    def get_memory_usage_mb(self) -> float:
        """Return current RSS memory in megabytes."""
        if self.process:
            try:
                return round(self.process.memory_info().rss / (1024 * 1024), 2)
            except Exception:
                return 0.0
        return 0.0

    def get_cpu_percent(self) -> float:
        """Return current process CPU percentage."""
        if self.process:
            try:
                return round(self.process.cpu_percent(interval=None), 2)
            except Exception:
                return 0.0
        return 0.0

    def measure_cold_vs_warm(
        self,
        func: Callable[[], Any],
        runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute func once for cold measurement, then multiple times for warm average.
        """
        # Cold start
        t0 = time.perf_counter()
        _ = func()
        cold_ms = (time.perf_counter() - t0) * 1000.0

        # Warm requests
        warm_timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = func()
            warm_timings.append((time.perf_counter() - t0) * 1000.0)

        warm_avg = sum(warm_timings) / max(len(warm_timings), 1)

        return {
            "cold_start_ms": round(cold_ms, 3),
            "warm_avg_ms": round(warm_avg, 3),
            "warm_min_ms": round(min(warm_timings), 3),
            "warm_max_ms": round(max(warm_timings), 3),
            "speedup_ratio": round(cold_ms / max(warm_avg, 0.001), 2),
            "memory_rss_mb": self.get_memory_usage_mb(),
        }
