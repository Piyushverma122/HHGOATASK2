import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from app.main import app
from voice.errors import (
    AudioValidationError,
    STTAuthenticationError,
    STTRateLimitError,
    STTTimeoutError,
    EmptyTranscriptError,
    STTProviderError,
)
from voice.audio.validator import AudioValidator
from voice.audio.preprocess import AudioPreprocessor
from voice.stt.sarvam import SarvamSTTProvider
from voice.stt.service import STTService
from voice.pipeline import VoicePipeline

client = TestClient(app)


class TestAudioValidation:
    """Test suite for audio format, size, duration, and stream validation."""

    def test_valid_synthetic_wav(self):
        validator = AudioValidator()
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=2.0)
        res = validator.validate_bytes(wav_bytes, filename="test.wav", mime_type="audio/wav")
        assert res["valid"] is True
        assert res["duration_seconds"] == 2.0
        assert res["sample_rate"] == 16000
        assert res["channels"] == 1

    def test_empty_audio_rejection(self):
        validator = AudioValidator()
        with pytest.raises(AudioValidationError) as exc:
            validator.validate_bytes(b"", filename="empty.wav")
        assert exc.value.code == "EMPTY_AUDIO"

    def test_unsupported_format_rejection(self):
        validator = AudioValidator()
        dummy_bytes = b"non-audio text content"
        with pytest.raises(AudioValidationError) as exc:
            validator.validate_bytes(dummy_bytes, filename="sample.txt", mime_type="text/plain")
        assert exc.value.code in ["UNSUPPORTED_MIME_TYPE", "UNSUPPORTED_AUDIO_FORMAT"]

    def test_oversized_audio_rejection(self):
        validator = AudioValidator(max_size_bytes=100)
        big_bytes = b"x" * 200
        with pytest.raises(AudioValidationError) as exc:
            validator.validate_bytes(big_bytes, filename="big.wav")
        assert exc.value.code == "AUDIO_TOO_LARGE"

    def test_audio_duration_limits(self):
        validator = AudioValidator(max_duration_seconds=5.0, min_duration_seconds=1.0)
        # Too long (6s)
        long_wav = AudioPreprocessor.create_synthetic_wav(duration_seconds=6.0)
        with pytest.raises(AudioValidationError) as exc_long:
            validator.validate_bytes(long_wav, filename="long.wav")
        assert exc_long.value.code == "AUDIO_TOO_LONG"


class TestSarvamSTTProvider:
    """Test suite for Sarvam AI STT provider, retries, and error handling."""

    def test_mock_transcription_mode(self):
        provider = SarvamSTTProvider(api_key="", mock_mode=True)
        assert provider.is_available() is True
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.5)
        res = provider.transcribe_bytes(wav_bytes, language_code="hi-IN")
        assert res.transcript == "भारत की राजधानी नई दिल्ली है।"
        assert res.language_code == "hi-IN"
        assert res.provider == "sarvam_mock"

    @patch("httpx.Client.post")
    def test_successful_sarvam_live_api_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "request_id": "sarvam_test_123",
            "transcript": "भारत की राजधानी क्या है?",
            "language_code": "hi-IN",
        }
        mock_post.return_value = mock_resp

        provider = SarvamSTTProvider(api_key="test_dummy_key", mock_mode=False)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.0)
        res = provider.transcribe_bytes(wav_bytes, language_code="hi", request_id="req_999")

        assert res.transcript == "भारत की राजधानी क्या है?"
        assert res.language_code == "hi-IN"
        assert res.request_id == "sarvam_test_123"

    @patch("httpx.Client.post")
    def test_sarvam_authentication_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid Subscription Key"
        mock_post.return_value = mock_resp

        provider = SarvamSTTProvider(api_key="invalid_key", mock_mode=False)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.0)
        with pytest.raises(STTAuthenticationError):
            provider.transcribe_bytes(wav_bytes)

    @patch("httpx.Client.post")
    def test_sarvam_rate_limit_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        mock_post.return_value = mock_resp

        provider = SarvamSTTProvider(api_key="key", mock_mode=False)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.0)
        with pytest.raises(STTRateLimitError):
            provider.transcribe_bytes(wav_bytes)

    @patch("httpx.Client.post")
    def test_sarvam_empty_transcript_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "request_id": "sarvam_test_123",
            "transcript": "   ",
            "language_code": "hi-IN",
        }
        mock_post.return_value = mock_resp

        provider = SarvamSTTProvider(api_key="key", mock_mode=False)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.0)
        with pytest.raises(EmptyTranscriptError):
            provider.transcribe_bytes(wav_bytes)

    @patch("httpx.Client.post")
    def test_sarvam_transient_retry_recovery(self, mock_post):
        # 1st attempt: 500 error, 2nd attempt: 200 OK success
        mock_err = MagicMock()
        mock_err.status_code = 500
        mock_err.text = "Internal Server Error"

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {
            "request_id": "sarvam_recovered",
            "transcript": "पुनः प्रयास सफल रहा।",
            "language_code": "hi-IN",
        }
        mock_post.side_effect = [mock_err, mock_ok]

        provider = SarvamSTTProvider(api_key="key", max_retries=3, backoff_factor=0.01, mock_mode=False)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.0)
        res = provider.transcribe_bytes(wav_bytes)
        assert res.transcript == "पुनः प्रयास सफल रहा।"


class TestVoicePipelineIntegration:
    """Test suite for Audio -> Sarvam STT -> Normalization -> Analysis -> Retrieval pipeline."""

    def test_full_voice_query_pipeline(self):
        stt_service = STTService(provider=SarvamSTTProvider(mock_mode=True))
        pipeline = VoicePipeline(stt_service=stt_service)
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=2.0)
        res = pipeline.process_voice_query(
            audio_bytes=wav_bytes,
            language_code="hi-IN",
            strategy="adaptive",
            rerank_top_k=5,
            enable_reranking=True,
            request_id="test_voice_trace_1",
        )

        assert "transcript" in res
        assert "normalized_query" in res
        assert "query_analysis" in res
        assert "final_context" in res
        assert "latency" in res
        assert res["latency"]["stt_ms"] >= 0
        assert res["latency"]["normalization_ms"] >= 0
        assert res["latency"]["total_ms"] > 0
        assert len(res["final_context"]) > 0


class TestVoiceAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    def test_get_voice_info_endpoint(self):
        resp = client.get("/api/v1/voice/info")
        assert resp.status_code == 200
        json_data = resp.json()
        assert json_data["success"] is True
        assert "provider_info" in json_data["data"]

    def test_transcribe_endpoint(self):
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.5)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"language": "hi-IN"}
        resp = client.post("/api/v1/voice/transcribe", files=files, data=data)
        assert resp.status_code == 200
        json_data = resp.json()
        assert json_data["success"] is True
        assert "transcript" in json_data["data"]
        assert "latency" in json_data["data"]

    def test_voice_query_endpoint(self):
        wav_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=1.5)
        files = {"file": ("recording.wav", wav_bytes, "audio/wav")}
        data = {"strategy": "adaptive", "language": "hi-IN", "top_k": 5}
        with patch(
            "voice.stt.service.STTService.transcribe_audio_bytes",
            return_value={
                "transcript": "भारत की राजधानी नई दिल्ली है।",
                "language_code": "hi-IN",
                "provider": "sarvam",
                "model": "saaras:v3",
                "duration_ms": 1500.0,
                "latency": {"stt_ms": 15.0, "total_ms": 20.0},
            },
        ):
            resp = client.post("/api/v1/voice/query", files=files, data=data)
            assert resp.status_code == 200
            json_data = resp.json()
            assert json_data["success"] is True
            assert "final_context" in json_data["data"]
            assert "latency" in json_data["data"]

    def test_text_query_fallback_endpoint(self):
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "strategy": "adaptive",
            "top_k": 5,
            "enable_reranking": True,
        }
        resp = client.post("/api/v1/voice/text-query", json=payload)
        assert resp.status_code == 200
        json_data = resp.json()
        assert json_data["success"] is True
        assert json_data["data"]["normalized_query"] == "भारत की राजधानी क्या है?"
        assert len(json_data["data"]["final_context"]) > 0
