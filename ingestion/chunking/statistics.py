import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from ingestion.chunking.models import Chunk


def calc_mean(values: List[Union[int, float]]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def calc_median(values: List[Union[int, float]]) -> float:
    if not values:
        return 0.0
    s_vals = sorted(values)
    n = len(s_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(s_vals[mid])
    else:
        return round((s_vals[mid - 1] + s_vals[mid]) / 2.0, 2)


class ChunkStatsCollector:
    """
    Collects execution metrics for a single chunking strategy run.
    """

    def __init__(self, strategy_name: str, sample_size: Optional[int] = None):
        self.strategy_name = strategy_name
        self.sample_size = sample_size
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None

        self.records_processed: int = 0
        self.passages_processed: int = 0
        self.chunks_generated: int = 0
        self.selected_passage_chunks: int = 0

        self.token_counts: List[int] = []
        self.char_counts: List[int] = []
        self.chunks_per_passage_list: List[int] = []
        self.chunks_per_query_list: List[int] = []

    def record_passage_result(self, chunks: List[Chunk]):
        self.passages_processed += 1
        self.chunks_per_passage_list.append(len(chunks))
        self.chunks_generated += len(chunks)

        for c in chunks:
            self.token_counts.append(c.token_count)
            self.char_counts.append(c.character_count)
            if c.is_selected_passage:
                self.selected_passage_chunks += 1

    def record_query_result(self, query_chunk_count: int):
        self.records_processed += 1
        self.chunks_per_query_list.append(query_chunk_count)

    def finalize(self) -> Dict[str, Any]:
        self.end_time = time.perf_counter()
        elapsed_seconds = round(self.end_time - self.start_time, 3)

        records_per_sec = round(self.records_processed / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0
        passages_per_sec = round(self.passages_processed / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0
        chunks_per_sec = round(self.chunks_generated / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0

        # Token count distribution buckets
        buckets = {
            "<64": 0,
            "64-128": 0,
            "128-256": 0,
            "256-384": 0,
            ">384": 0,
        }
        for tc in self.token_counts:
            if tc < 64:
                buckets["<64"] += 1
            elif tc <= 128:
                buckets["64-128"] += 1
            elif tc <= 256:
                buckets["128-256"] += 1
            elif tc <= 384:
                buckets["256-384"] += 1
            else:
                buckets[">384"] += 1

        return {
            "strategy": self.strategy_name,
            "sample_size": self.sample_size,
            "records_processed": self.records_processed,
            "passages_processed": self.passages_processed,
            "chunks_generated": self.chunks_generated,
            "selected_passage_chunks": self.selected_passage_chunks,
            "ratios": {
                "avg_chunks_per_passage": calc_mean(self.chunks_per_passage_list),
                "avg_chunks_per_query": calc_mean(self.chunks_per_query_list),
            },
            "tokens": {
                "average": calc_mean(self.token_counts),
                "median": calc_median(self.token_counts),
                "min": min(self.token_counts) if self.token_counts else 0,
                "max": max(self.token_counts) if self.token_counts else 0,
                "distribution_buckets": buckets,
            },
            "characters": {
                "average": calc_mean(self.char_counts),
                "median": calc_median(self.char_counts),
                "min": min(self.char_counts) if self.char_counts else 0,
                "max": max(self.char_counts) if self.char_counts else 0,
            },
            "performance": {
                "elapsed_seconds": elapsed_seconds,
                "records_per_sec": records_per_sec,
                "passages_per_sec": passages_per_sec,
                "chunks_per_sec": chunks_per_sec,
            },
        }


def generate_comparison_report(
    all_strategy_stats: List[Dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Generate structured comparison reports across all evaluated strategies.
    Outputs:
    - data/statistics/chunking_comparison.json
    - data/statistics/chunking_comparison.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "chunking_comparison.json"
    md_path = output_dir / "chunking_comparison.md"

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategies_compared": len(all_strategy_stats),
        "results": all_strategy_stats,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    # Build Markdown Comparison Table
    table_rows = []
    for s in all_strategy_stats:
        table_rows.append(
            f"| **{s['strategy'].capitalize()}** | {s['chunks_generated']:,} | "
            f"{s['ratios']['avg_chunks_per_passage']} | {s['tokens']['average']} | "
            f"{s['tokens']['median']} | {s['tokens']['min']} - {s['tokens']['max']} | "
            f"{s['selected_passage_chunks']:,} | {s['performance']['elapsed_seconds']}s | "
            f"{s['performance']['chunks_per_sec']} |"
        )

    rows_str = "\n".join(table_rows)

    md_content = f"""# Multi-Strategy Chunking Comparison Report

**HH Goa 2026 — Task 2 | Module 3: Advanced Multi-Strategy Chunking**

---

## 1. Strategy Comparison Matrix

| Strategy | Total Chunks | Avg Chunks/Passage | Avg Tokens | Median Tokens | Token Range | Selected Chunks | Processing Time | Chunks/Sec |
|---|---|---|---|---|---|---|---|---|
{rows_str}

---

## 2. Key Insights & Strategy Trade-Offs

- **Fixed-Size**: Simple baseline, consistent token size but cuts through sentences and Indic words arbitrarily.
- **Fixed + Overlap**: Retains context across boundaries via sliding window, increasing total chunk volume.
- **Sentence-Aware**: Eliminates broken sentence fragments; preserves complete Hindi sentences (respecting `।`, `॥`, `?`).
- **Paragraph-Aware**: Preserves document structure for multi-paragraph passages, using sentence fallback when oversized.
- **Semantic Chunking**: Computes sentence similarities and creates boundaries at topic shift valleys.
- **Metadata-Aware**: Adapts granularity according to `query_type` (fine-grained for `NUMERIC`, entity-focused for `ENTITY`/`PERSON`/`LOCATION`).
- **Adaptive Chunking**: Deterministic decision tree routing each passage to optimal strategy based on passage length, structural density, and query type.

---

## 3. Token Distribution Breakdown

```json
{json.dumps({s['strategy']: s['tokens']['distribution_buckets'] for s in all_strategy_stats}, indent=2)}
```
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path
