import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pyarrow as pa
import pyarrow.parquet as pq


# Define PyArrow schema for canonical records
PASSAGE_STRUCT_TYPE = pa.struct([
    ("passage_id", pa.string()),
    ("passage_index", pa.int32()),
    ("text", pa.string()),
    ("english_text", pa.string()),
    ("is_selected", pa.bool_()),
])

CANONICAL_PYARROW_SCHEMA = pa.schema([
    ("record_id", pa.string()),
    ("query_id", pa.int64()),
    ("query", pa.string()),
    ("answer", pa.string()),
    ("query_type", pa.string()),
    ("source_lang", pa.string()),
    ("target_lang", pa.string()),
    ("passages", pa.list_(PASSAGE_STRUCT_TYPE)),
    ("original_eng_query", pa.string()),
    ("original_eng_answer", pa.string()),
])


class ParquetBatchWriter:
    """
    Incremental batch writer for Parquet files using PyArrow.
    Streams batches directly to disk without loading entire dataset into memory.
    """

    def __init__(self, output_path: Path, schema: pa.Schema = CANONICAL_PYARROW_SCHEMA):
        self.output_path = output_path
        self.schema = schema
        self.writer: Optional[pq.ParquetWriter] = None
        self.total_records_written: int = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_batch(self, records: List[Dict[str, Any]]):
        if not records:
            return

        # Prepare column arrays
        record_ids = []
        query_ids = []
        queries = []
        answers = []
        query_types = []
        source_langs = []
        target_langs = []
        passages_list = []
        eng_queries = []
        eng_answers = []

        for r in records:
            record_ids.append(str(r.get("record_id", "")))
            q_id = r.get("query_id")
            query_ids.append(int(q_id) if q_id is not None else 0)
            queries.append(r.get("query", ""))
            answers.append(r.get("answer") or "")
            query_types.append(r.get("query_type") or "")
            source_langs.append(r.get("source_lang") or "")
            target_langs.append(r.get("target_lang") or "")

            raw_passages = r.get("passages", [])
            formatted_passages = []
            for p in raw_passages:
                formatted_passages.append({
                    "passage_id": str(p.get("passage_id", "")),
                    "passage_index": int(p.get("passage_index", 0)),
                    "text": p.get("text", "") or "",
                    "english_text": p.get("english_text", "") or "",
                    "is_selected": bool(p.get("is_selected", False)),
                })
            passages_list.append(formatted_passages)

            orig = r.get("original", {})
            eng_queries.append(orig.get("eng_query") or "")
            eng_answers.append(orig.get("eng_answer") or "")

        data = {
            "record_id": record_ids,
            "query_id": query_ids,
            "query": queries,
            "answer": answers,
            "query_type": query_types,
            "source_lang": source_langs,
            "target_lang": target_langs,
            "passages": passages_list,
            "original_eng_query": eng_queries,
            "original_eng_answer": eng_answers,
        }

        table = pa.Table.from_pydict(data, schema=self.schema)

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                str(self.output_path),
                self.schema,
                compression="snappy",
            )

        self.writer.write_table(table)
        self.total_records_written += len(records)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def export_jsonl(records: List[Dict[str, Any]], output_path: Path):
    """Export records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
