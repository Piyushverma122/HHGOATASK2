import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ingestion.dataset_loader import (
    LANGUAGE_CODE_MAP,
    LANGUAGE_NAMES,
    stream_raw_records,
    raw_to_canonical,
)
from ingestion.config import ingestion_settings


def inspect_dataset_info(dataset_name: str = "ai4bharat/MSMARCO-XI", sample_lang: str = "hi") -> Dict[str, Any]:
    """
    Programmatically inspects the remote Hugging Face dataset configuration, splits, features, and sample schemas.
    """
    print("=" * 60)
    print(f"DATASET INSPECTION REPORT: {dataset_name}")
    print("=" * 60)

    # 1. Configurations (Languages)
    configs = list(LANGUAGE_CODE_MAP.keys())
    print(f"\nAvailable Language Configurations ({len(configs)}):")
    for code in configs:
        print(f"  - {code}: {LANGUAGE_NAMES[code]} (File prefix: {LANGUAGE_CODE_MAP[code]})")

    # 2. Splits
    splits = ["train", "validation"]
    print(f"\nAvailable Dataset Splits:")
    for s in splits:
        print(f"  - {s}")

    # 3. Features & Schema
    print("\nDataset Features / Schema Definition:")
    schema_fields = {
        "source_lang": "string (e.g. 'eng_Latn')",
        "target_lang": "string (e.g. 'hin_Deva')",
        "meta": "struct {model_name, temperature, max_tokens, top_p, frequency_penalty, presence_penalty}",
        "query": "string (Translated Indic query)",
        "Answer": "string (Translated Indic answer)",
        "query_id": "int32 / int64",
        "query_type": "string (e.g. 'DESCRIPTION', 'ENTITY', 'NUMERIC')",
        "passages": "struct {is_selected: list[int32], English_passages: list[string], Translated_passages: list[string]}",
        "Eng_Query": "string (Original English query)",
        "Eng_Answer": "string (Original English answer)",
    }
    for k, v in schema_fields.items():
        print(f"  - {k}: {v}")

    # 4. Stream 1 sample record to verify actual structure
    print(f"\nFetching 1 Live Sample Record ({sample_lang} - validation)...")
    sample_raw = None
    try:
        stream = stream_raw_records(language=sample_lang, split="validation", sample_size=1)
        for item in stream:
            sample_raw = item
            break
    except Exception as e:
        print(f"Error fetching live sample: {e}")

    if sample_raw:
        canonical_sample = raw_to_canonical(sample_raw, language=sample_lang)
        print("\n--- Raw Sample Keys ---")
        print(list(sample_raw.keys()))

        print("\n--- Canonical Processed Record Sample ---")
        print(json.dumps(canonical_sample, indent=2, ensure_ascii=False))

    report = {
        "dataset_name": dataset_name,
        "configurations": configs,
        "splits": splits,
        "schema_fields": schema_fields,
        "sample_raw_keys": list(sample_raw.keys()) if sample_raw else [],
    }

    return report


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
    inspect_dataset_info(sample_lang=lang)
