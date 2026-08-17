import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("voice_rag.benchmark.fixtures")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "voice"


def load_voice_fixtures() -> List[Dict[str, Any]]:
    """
    Load all pre-recorded / verified voice STT fixtures from data/fixtures/voice/.
    Guarantees zero Sarvam API quota consumption during benchmark evaluations.
    """
    if not FIXTURES_DIR.exists():
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        return []

    fixtures = []
    for json_file in sorted(FIXTURES_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["fixture_file"] = json_file.name
                fixtures.append(data)
        except Exception as e:
            logger.error(f"Failed to load fixture {json_file}: {e}")

    return fixtures


def get_fixture_by_language(lang: str) -> Optional[Dict[str, Any]]:
    """Get a specific language fixture."""
    fixtures = load_voice_fixtures()
    for f in fixtures:
        if f.get("language") == lang or f.get("language_code", "").startswith(lang):
            return f
    return fixtures[0] if fixtures else None
