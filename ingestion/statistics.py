import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


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


class IngestionStatsCollector:
    """
    Collects, aggregates, and exports ingestion statistics.
    Computes exact statistical metrics (mean, median, min, max, distributions).
    """

    def __init__(self, dataset_name: str, language: str, split: str, sample_size: Optional[int] = None):
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        self.sample_size = sample_size
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None

        # Rows
        self.rows_processed: int = 0
        self.rows_valid: int = 0
        self.rows_invalid: int = 0

        # Queries
        self.query_types: Dict[str, int] = {}
        self.query_lengths: List[int] = []

        # Answers
        self.answer_lengths: List[int] = []
        self.empty_answers_count: int = 0

        # Passages
        self.total_passages: int = 0
        self.selected_passages: int = 0
        self.passage_lengths: List[int] = []
        self.passages_per_query: List[int] = []

        # Errors & Duplicates
        self.error_counts: Dict[str, int] = {}
        self.duplicate_records_count: int = 0
        self.unique_queries_count: int = 0

    def record_row(
        self,
        is_valid: bool,
        is_duplicate: bool,
        record: Optional[Dict[str, Any]] = None,
        errors: Optional[List[Any]] = None,
    ):
        self.rows_processed += 1

        if is_duplicate:
            self.duplicate_records_count += 1
            return

        if not is_valid:
            self.rows_invalid += 1
            if errors:
                for err in errors:
                    code = getattr(err, "error_code", str(err))
                    self.error_counts[code] = self.error_counts.get(code, 0) + 1
            return

        self.rows_valid += 1
        self.unique_queries_count += 1

        if record:
            # Query stats
            query = record.get("query", "")
            self.query_lengths.append(len(query))

            q_type = record.get("query_type") or "standard"
            self.query_types[q_type] = self.query_types.get(q_type, 0) + 1

            # Answer stats
            answer = record.get("answer", "")
            if answer:
                self.answer_lengths.append(len(answer))
            else:
                self.empty_answers_count += 1

            # Passage stats
            passages = record.get("passages", [])
            self.passages_per_query.append(len(passages))
            for p in passages:
                self.total_passages += 1
                if p.get("is_selected", False):
                    self.selected_passages += 1
                text = p.get("text", "")
                if text:
                    self.passage_lengths.append(len(text))

    def finalize(self) -> Dict[str, Any]:
        self.end_time = time.perf_counter()
        elapsed_seconds = round(self.end_time - self.start_time, 3)
        records_per_sec = round(self.rows_processed / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0

        avg_query_len = calc_mean(self.query_lengths)
        median_query_len = calc_median(self.query_lengths)

        avg_ans_len = calc_mean(self.answer_lengths)
        median_ans_len = calc_median(self.answer_lengths)

        avg_p_count = calc_mean(self.passages_per_query)
        avg_p_len = calc_mean(self.passage_lengths)
        median_p_len = calc_median(self.passage_lengths)
        min_p_len = min(self.passage_lengths) if self.passage_lengths else 0
        max_p_len = max(self.passage_lengths) if self.passage_lengths else 0

        stats: Dict[str, Any] = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "language": self.language,
                "split": self.split,
                "sample_size_requested": self.sample_size,
                "elapsed_seconds": elapsed_seconds,
                "throughput_records_per_sec": records_per_sec,
            },
            "rows": {
                "total_processed": self.rows_processed,
                "valid": self.rows_valid,
                "invalid": self.rows_invalid,
                "duplicates": self.duplicate_records_count,
            },
            "queries": {
                "total_queries": self.rows_valid,
                "unique_queries": self.unique_queries_count,
                "query_types": self.query_types,
                "length": {
                    "average_chars": avg_query_len,
                    "median_chars": median_query_len,
                    "min_chars": min(self.query_lengths) if self.query_lengths else 0,
                    "max_chars": max(self.query_lengths) if self.query_lengths else 0,
                },
            },
            "answers": {
                "total_with_answer": len(self.answer_lengths),
                "empty_answers": self.empty_answers_count,
                "length": {
                    "average_chars": avg_ans_len,
                    "median_chars": median_ans_len,
                },
            },
            "passages": {
                "total_passages": self.total_passages,
                "selected_passages": self.selected_passages,
                "average_passages_per_query": avg_p_count,
                "length": {
                    "average_chars": avg_p_len,
                    "median_chars": median_p_len,
                    "min_chars": min_p_len,
                    "max_chars": max_p_len,
                },
            },
            "errors": {
                "error_breakdown": self.error_counts,
                "total_error_events": sum(self.error_counts.values()),
            },
        }

        return stats

    def export_reports(self, output_dir: Path) -> tuple[Path, Path]:
        stats = self.finalize()
        json_filename = f"dataset_stats_{self.language}_{self.split}.json"
        md_filename = f"dataset_stats_{self.language}_{self.split}.md"

        json_path = output_dir / json_filename
        md_path = output_dir / md_filename

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        md_content = f"""# MSMARCO-XI Ingestion Report ({self.language.upper()} - {self.split})

## Dataset Metadata
- **Dataset Name**: `{stats['metadata']['dataset_name']}`
- **Language**: `{stats['metadata']['language']}`
- **Split**: `{stats['metadata']['split']}`
- **Requested Sample Size**: `{stats['metadata']['sample_size_requested'] or 'Full'}`
- **Processing Time**: `{stats['metadata']['elapsed_seconds']}s`
- **Throughput**: `{stats['metadata']['throughput_records_per_sec']} records/sec`

## Processing Summary
| Metric | Value |
|---|---|
| Total Processed | {stats['rows']['total_processed']:,} |
| Valid Records | {stats['rows']['valid']:,} |
| Invalid Records | {stats['rows']['invalid']:,} |
| Duplicate Records | {stats['rows']['duplicates']:,} |

## Query Statistics
- **Total Valid Queries**: {stats['queries']['total_queries']:,}
- **Average Query Length**: {stats['queries']['length']['average_chars']} characters
- **Median Query Length**: {stats['queries']['length']['median_chars']} characters
- **Query Length Range**: {stats['queries']['length']['min_chars']} - {stats['queries']['length']['max_chars']} characters
- **Query Types Breakdown**:
```json
{json.dumps(stats['queries']['query_types'], indent=2)}
```

## Passage Statistics
- **Total Passages**: {stats['passages']['total_passages']:,}
- **Selected Passages (Ground Truth)**: {stats['passages']['selected_passages']:,}
- **Avg Passages / Query**: {stats['passages']['average_passages_per_query']}
- **Average Passage Length**: {stats['passages']['length']['average_chars']} characters
- **Median Passage Length**: {stats['passages']['length']['median_chars']} characters
- **Min / Max Passage Length**: {stats['passages']['length']['min_chars']} / {stats['passages']['length']['max_chars']} characters

## Validation & Errors
- **Empty Answers**: {stats['answers']['empty_answers']:,}
- **Validation Errors Breakdown**:
```json
{json.dumps(stats['errors']['error_breakdown'], indent=2)}
```
"""

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path
