from typing import Any, Dict, List, Tuple, Optional
from ingestion.normalize import is_empty_text


class RecordValidationError:
    """Encapsulates a validation failure reason."""

    def __init__(self, record_id: Any, error_code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.record_id = record_id
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


def validate_canonical_record(record: Dict[str, Any]) -> Tuple[bool, List[RecordValidationError]]:
    """
    Validate a canonical MSMARCO-XI record.
    Returns (is_valid, list_of_errors).
    """
    errors: List[RecordValidationError] = []
    record_id = record.get("record_id") or record.get("query_id") or "UNKNOWN"

    # 1. Validate Query
    query = record.get("query")
    if is_empty_text(query):
        errors.append(
            RecordValidationError(
                record_id=record_id,
                error_code="EMPTY_QUERY",
                message="Query text is missing or empty after normalization",
            )
        )

    # 2. Validate Query ID
    query_id = record.get("query_id")
    if query_id is None:
        errors.append(
            RecordValidationError(
                record_id=record_id,
                error_code="MISSING_QUERY_ID",
                message="Query ID is missing",
            )
        )

    # 3. Validate Passages
    passages = record.get("passages")
    if not isinstance(passages, list) or len(passages) == 0:
        errors.append(
            RecordValidationError(
                record_id=record_id,
                error_code="EMPTY_PASSAGES",
                message="Record has no associated passages",
            )
        )
    else:
        valid_passage_found = False
        for idx, p in enumerate(passages):
            if not isinstance(p, dict):
                errors.append(
                    RecordValidationError(
                        record_id=record_id,
                        error_code="INVALID_PASSAGE_TYPE",
                        message=f"Passage at index {idx} is not a valid object",
                    )
                )
                continue

            text = p.get("text")
            if not is_empty_text(text):
                valid_passage_found = True

        if not valid_passage_found:
            errors.append(
                RecordValidationError(
                    record_id=record_id,
                    error_code="ALL_PASSAGES_EMPTY",
                    message="All passages in record are empty",
                )
            )

    # 4. Validate Language Metadata
    target_lang = record.get("target_lang")
    if not target_lang or not isinstance(target_lang, str):
        errors.append(
            RecordValidationError(
                record_id=record_id,
                error_code="INVALID_LANGUAGE",
                message="Target language metadata is missing or invalid",
            )
        )

    is_valid = len(errors) == 0
    return is_valid, errors
