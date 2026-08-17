import pytest
from pathlib import Path
from ingestion.normalize import normalize_text, is_empty_text
from ingestion.validate import validate_canonical_record
from ingestion.deduplicate import Deduplicator
from ingestion.statistics import IngestionStatsCollector
from ingestion.dataset_loader import raw_to_canonical, get_hf_filename
from ingestion.export import ParquetBatchWriter, export_jsonl
import pyarrow.parquet as pq


def test_unicode_normalization_and_indic_preservation():
    # Hindi text with matras, halant, virama, and punctuation
    raw_hindi = "  क्या  यह   भारत  की \t राजधानी \r\n है?   "
    normalized = normalize_text(raw_hindi)
    assert normalized == "क्या यह भारत की राजधानी है?"

    # Bengali text with virama and punctuation
    raw_bengali = "কলকাতা\u200c পশ্চিমবঙ্গ\r\n"
    assert normalize_text(raw_bengali) == "কলকাতা\u200c পশ্চিমবঙ্গ"

    # Control characters removal (excluding valid spaces/tabs/newlines)
    control_text = "Hello\x00\x07World\nTest"
    assert normalize_text(control_text) == "HelloWorld Test"


def test_empty_text_detection():
    assert is_empty_text("") is True
    assert is_empty_text("   \n\t  ") is True
    assert is_empty_text(None) is True
    assert is_empty_text("valid text") is False


def test_raw_to_canonical_and_stable_passage_ids():
    raw_item = {
        "query_id": 1001,
        "query": "भारत की राजधानी क्या है?",
        "Answer": "नई दिल्ली",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {"model_name": "indic-trans"},
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["New Delhi is the capital.", "Mumbai is financial hub."],
            "Translated_passages": ["नई दिल्ली राजधानी है।", "मुंबई वित्तीय केंद्र है।"],
        },
        "Eng_Query": "What is capital of India?",
        "Eng_Answer": "New Delhi",
    }

    canonical = raw_to_canonical(raw_item, language="hi")

    assert canonical["record_id"] == "1001_hi"
    assert canonical["query_id"] == 1001
    assert canonical["query"] == "भारत की राजधानी क्या है?"
    assert canonical["answer"] == "नई दिल्ली"
    assert canonical["source_lang"] == "eng_Latn"
    assert canonical["target_lang"] == "hin_Deva"
    assert len(canonical["passages"]) == 2

    # Verify deterministic stable passage IDs
    p0 = canonical["passages"][0]
    p1 = canonical["passages"][1]
    assert p0["passage_id"] == "1001_hi_0"
    assert p0["is_selected"] is True
    assert p0["text"] == "नई दिल्ली राजधानी है।"
    assert p0["english_text"] == "New Delhi is the capital."

    assert p1["passage_id"] == "1001_hi_1"
    assert p1["is_selected"] is False


def test_validation_logic():
    # Valid record
    valid_rec = {
        "record_id": "1_hi",
        "query_id": 1,
        "query": "valid query",
        "target_lang": "hin_Deva",
        "passages": [{"text": "valid passage", "is_selected": True}],
    }
    is_valid, errors = validate_canonical_record(valid_rec)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid record (empty query and missing passages)
    invalid_rec = {
        "record_id": "2_hi",
        "query_id": 2,
        "query": "   ",
        "target_lang": "hin_Deva",
        "passages": [],
    }
    is_valid, errors = validate_canonical_record(invalid_rec)
    assert is_valid is False
    error_codes = [e.error_code for e in errors]
    assert "EMPTY_QUERY" in error_codes
    assert "EMPTY_PASSAGES" in error_codes


def test_deduplication():
    dedup = Deduplicator()

    rec1 = {"query_id": 101, "query": "Query One"}
    rec2 = {"query_id": 101, "query": "Query One Duplicate"}
    rec3 = {"query_id": 102, "query": "Query One"}  # Same text, different ID
    rec4 = {"query_id": 103, "query": "Unique Query"}

    is_unique1, reason1 = dedup.process_record(rec1)
    is_unique2, reason2 = dedup.process_record(rec2)
    is_unique3, reason3 = dedup.process_record(rec3)
    is_unique4, reason4 = dedup.process_record(rec4)

    assert is_unique1 is True
    assert is_unique2 is False
    assert "DUPLICATE_QUERY_ID" in reason2
    assert is_unique3 is False
    assert "DUPLICATE_QUERY_TEXT" in reason3
    assert is_unique4 is True

    stats = dedup.get_stats()
    assert stats["total_records"] == 4
    assert stats["duplicate_records"] == 2
    assert stats["unique_records"] == 2


def test_statistics_aggregation_and_reporting(tmp_path):
    collector = IngestionStatsCollector("ai4bharat/MSMARCO-XI", "hi", "train", sample_size=10)

    sample_rec = {
        "record_id": "1_hi",
        "query_id": 1,
        "query": "Query text",
        "answer": "Answer text",
        "query_type": "standard",
        "passages": [
            {"text": "Passage 1 text", "is_selected": True},
            {"text": "Passage 2 longer text", "is_selected": False},
        ],
    }

    collector.record_row(is_valid=True, is_duplicate=False, record=sample_rec)
    collector.record_row(is_valid=False, is_duplicate=False, errors=["EMPTY_QUERY"])
    collector.record_row(is_valid=False, is_duplicate=True)

    json_path, md_path = collector.export_reports(tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    stats = collector.finalize()
    assert stats["rows"]["total_processed"] == 3
    assert stats["rows"]["valid"] == 1
    assert stats["rows"]["invalid"] == 1
    assert stats["rows"]["duplicates"] == 1
    assert stats["passages"]["total_passages"] == 2
    assert stats["passages"]["selected_passages"] == 1


def test_parquet_batch_writer(tmp_path):
    out_file = tmp_path / "test_out.parquet"
    writer = ParquetBatchWriter(out_file)

    records = [
        {
            "record_id": "1_hi",
            "query_id": 1,
            "query": "Test query 1",
            "answer": "Answer 1",
            "query_type": "DESCRIPTION",
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "passages": [
                {
                    "passage_id": "1_hi_0",
                    "passage_index": 0,
                    "text": "Passage 1",
                    "english_text": "Eng 1",
                    "is_selected": True,
                }
            ],
            "original": {"eng_query": "Eng Q", "eng_answer": "Eng A"},
        }
    ]

    writer.write_batch(records)
    writer.close()

    assert out_file.exists()
    table = pq.read_table(out_file)
    assert table.num_rows == 1
    assert table.column("record_id")[0].as_py() == "1_hi"
    assert table.column("query")[0].as_py() == "Test query 1"


def test_language_code_mapping():
    assert get_hf_filename("hi", "train") == "train/hintrain.parquet"
    assert get_hf_filename("ta", "validation") == "validation/tamval.parquet"
    assert get_hf_filename("bn", "val") == "validation/benval.parquet"

    with pytest.raises(ValueError):
        get_hf_filename("unsupported_lang", "train")
