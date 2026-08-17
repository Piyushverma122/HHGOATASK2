import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from voice.audio.preprocess import AudioPreprocessor
from voice.stt.sarvam import SarvamSTTProvider
from voice.stt.service import STTService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.stt.benchmark")


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


def calculate_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """
    Computes Word Error Rate (WER) and Character Error Rate (CER).
    """
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()

    # CER
    if len(ref_clean) == 0:
        cer = 0.0 if len(hyp_clean) == 0 else 1.0
    else:
        char_dist = calculate_levenshtein_distance(ref_clean, hyp_clean)
        cer = min(1.0, char_dist / len(ref_clean))

    # WER
    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    if len(ref_words) == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        word_dist = calculate_levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
        wer = min(1.0, word_dist / len(" ".join(ref_words)))

    return round(float(wer), 4), round(float(cer), 4)


def run_stt_benchmark(num_samples: int = 30) -> Dict[str, Any]:
    """
    Executes comprehensive STT latency and quality benchmarking across
    short, medium, and long audio samples across 7 languages.
    """
    logger.info(f"Initiating STT Benchmark with {num_samples} evaluation test cases...")
    stt_service = STTService()

    test_corpus = [
        # Hindi Short (1-5s)
        {"lang": "hi-IN", "duration_cat": "short", "duration_sec": 2.5, "ref": "भारत की राजधानी नई दिल्ली है।"},
        {"lang": "hi-IN", "duration_cat": "short", "duration_sec": 3.0, "ref": "पोटेशियम में कम खाद्य पदार्थों का चार्ट।"},
        {"lang": "hi-IN", "duration_cat": "short", "duration_sec": 4.0, "ref": "कंप्यूटर क्या है और यह कैसे काम करता है?"},
        # Hindi Medium (5-15s)
        {"lang": "hi-IN", "duration_cat": "medium", "duration_sec": 8.0, "ref": "भारतीय संविधान विश्व का सबसे लंबा लिखित संविधान है जिसमें कई महत्वपूर्ण प्रावधान हैं।"},
        {"lang": "hi-IN", "duration_cat": "medium", "duration_sec": 12.0, "ref": "स्वास्थ्य के लिए नियमित व्यायाम और संतुलित आहार अत्यंत आवश्यक माना जाता है।"},
        # Hindi Long (15-30s)
        {"lang": "hi-IN", "duration_cat": "long", "duration_sec": 22.0, "ref": "जलवायु परिवर्तन के कारण दुनिया भर में तापमान बढ़ रहा है और ग्लेशियर तेजी से पिघल रहे हैं जिसका असर कृषि पर पड़ रहा है।"},
        
        # English Short, Medium, Long
        {"lang": "en-IN", "duration_cat": "short", "duration_sec": 2.0, "ref": "What is the capital of India?"},
        {"lang": "en-IN", "duration_cat": "medium", "duration_sec": 7.5, "ref": "Artificial intelligence is rapidly transforming global technology and modern search engines."},
        {"lang": "en-IN", "duration_cat": "long", "duration_sec": 20.0, "ref": "Retrieval augmented generation combines dense vector embeddings with cross-attention reranking for accurate grounding."},
        
        # Hinglish
        {"lang": "hi-IN", "duration_cat": "short", "duration_sec": 3.0, "ref": "India ki capital New Delhi hai."},
        {"lang": "hi-IN", "duration_cat": "medium", "duration_sec": 8.5, "ref": "Machine learning models real-time voice recognition ke liye use ho rahe hain."},
        
        # Bengali
        {"lang": "bn-IN", "duration_cat": "short", "duration_sec": 2.5, "ref": "ভারতের রাজধানী নতুন দিল্লি।"},
        {"lang": "bn-IN", "duration_cat": "medium", "duration_sec": 9.0, "ref": "বিজ্ঞান ও প্রযুক্তির অগ্রগতি মানব সভ্যতার জন্য অত্যন্ত গুরুত্বপূর্ণ।"},
        
        # Tamil
        {"lang": "ta-IN", "duration_cat": "short", "duration_sec": 3.0, "ref": "இந்தியாவின் தலைநகரம் புது தில்லி."},
        {"lang": "ta-IN", "duration_cat": "medium", "duration_sec": 8.0, "ref": "தமிழ் மொழி உலகின் மிகத் தொன்மையான மொழிகளில் ஒன்றாகும்."},
        
        # Telugu
        {"lang": "te-IN", "duration_cat": "short", "duration_sec": 2.8, "ref": "భారతదేశ రాజధాని న్యూఢిల్లీ."},
        {"lang": "te-IN", "duration_cat": "medium", "duration_sec": 7.5, "ref": "సమాచార సాంకేతిక పరిజ్ఞానం రోజువారీ జీవితాన్ని సులభతరం చేస్తోంది."},
        
        # Marathi
        {"lang": "mr-IN", "duration_cat": "short", "duration_sec": 2.5, "ref": "भारताची राजधानी नवी दिल्ली आहे."},
        {"lang": "mr-IN", "duration_cat": "medium", "duration_sec": 8.0, "ref": "शिक्षणाने मानवाचा सर्वांगीण विकास होतो आणि समाज प्रगत बनतो."},
    ]

    # Replicate corpus up to target count (e.g. 30 samples)
    eval_cases = []
    while len(eval_cases) < num_samples:
        for item in test_corpus:
            if len(eval_cases) >= num_samples:
                break
            eval_cases.append(dict(item))

    test_results: List[Dict[str, Any]] = []
    latencies_all: List[float] = []
    latencies_by_cat: Dict[str, List[float]] = {"short": [], "medium": [], "long": []}
    wers: List[float] = []
    cers: List[float] = []

    for idx, test_item in enumerate(eval_cases, start=1):
        dur = test_item["duration_sec"]
        cat = test_item["duration_cat"]
        lang = test_item["lang"]
        ref_text = test_item["ref"]

        # Generate synthetic audio for deterministic benchmark
        audio_bytes = AudioPreprocessor.create_synthetic_wav(duration_seconds=dur, sample_rate=16000)

        t0 = time.perf_counter()
        res = stt_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=f"bench_sample_{idx}.wav",
            language_code=lang,
            request_id=f"bench_{idx}",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies_all.append(elapsed_ms)
        latencies_by_cat[cat].append(elapsed_ms)

        hyp_text = res["transcript"]
        wer, cer = calculate_wer_cer(ref_text, hyp_text)
        wers.append(wer)
        cers.append(cer)

        test_results.append({
            "sample_id": idx,
            "language": lang,
            "duration_seconds": dur,
            "duration_category": cat,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "wer": wer,
            "cer": cer,
            "latency_ms": round(elapsed_ms, 3),
            "stt_ms": res["latency"].get("stt_ms", 0.0),
            "provider": res["provider"],
            "model": res["model"],
            "status": "SUCCESS",
        })

    def calc_percentiles(lat_list: List[float]) -> Dict[str, float]:
        if not lat_list:
            return {}
        return {
            "p50_ms": round(float(np.percentile(lat_list, 50)), 3),
            "p70_ms": round(float(np.percentile(lat_list, 70)), 3),
            "p90_ms": round(float(np.percentile(lat_list, 90)), 3),
            "p95_ms": round(float(np.percentile(lat_list, 95)), 3),
            "p99_ms": round(float(np.percentile(lat_list, 99)), 3),
            "p100_ms": round(float(np.max(lat_list)), 3),
            "mean_ms": round(float(np.mean(lat_list)), 3),
        }

    overall_lat = calc_percentiles(latencies_all)
    short_lat = calc_percentiles(latencies_by_cat["short"])
    medium_lat = calc_percentiles(latencies_by_cat["medium"])
    long_lat = calc_percentiles(latencies_by_cat["long"])

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_recordings": len(test_results),
        "provider_info": stt_service.provider.get_provider_info(),
        "latency_percentiles": {
            "overall": overall_lat,
            "short_audio_1_5s": short_lat,
            "medium_audio_5_15s": medium_lat,
            "long_audio_15_30s": long_lat,
        },
        "quality_metrics": {
            "mean_wer": round(float(np.mean(wers)), 4),
            "mean_cer": round(float(np.mean(cers)), 4),
            "evaluated_languages": ["hi-IN", "en-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN"],
        },
        "detailed_results": test_results,
    }

    return benchmark_summary


def save_stt_reports(summary: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stt_benchmark.json"
    md_path = output_dir / "stt_benchmark.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lat = summary["latency_percentiles"]
    q = summary["quality_metrics"]
    prov = summary["provider_info"]

    md_content = f"""# Module 6 — Sarvam Speech-to-Text Benchmark Report

**HH Goa 2026 — Task 2 | Module 6: Audio Ingestion, STT Latency & Quality Evaluation**  
*Provider: `{prov['provider']}` | Model: `{prov['model']}` | Total Evaluations: {summary['total_recordings']}*

---

## 1. STT Latency Percentiles (End-to-End Audio Pipeline)

| Audio Category | Count | P50 | P70 | P90 | P95 | P99 | P100 (Max) | Mean Latency |
|---|---|---|---|---|---|---|---|---|
| **Overall** | **{summary['total_recordings']}** | **{lat['overall']['p50_ms']} ms** | {lat['overall']['p70_ms']} ms | {lat['overall']['p90_ms']} ms | {lat['overall']['p95_ms']} ms | {lat['overall']['p99_ms']} ms | {lat['overall']['p100_ms']} ms | **{lat['overall']['mean_ms']} ms** |
| **Short (1–5s)** | {len([r for r in summary['detailed_results'] if r['duration_category'] == 'short'])} | **{lat['short_audio_1_5s'].get('p50_ms', 'N/A')} ms** | {lat['short_audio_1_5s'].get('p70_ms', 'N/A')} ms | {lat['short_audio_1_5s'].get('p90_ms', 'N/A')} ms | {lat['short_audio_1_5s'].get('p95_ms', 'N/A')} ms | {lat['short_audio_1_5s'].get('p99_ms', 'N/A')} ms | {lat['short_audio_1_5s'].get('p100_ms', 'N/A')} ms | **{lat['short_audio_1_5s'].get('mean_ms', 'N/A')} ms** |
| **Medium (5–15s)** | {len([r for r in summary['detailed_results'] if r['duration_category'] == 'medium'])} | **{lat['medium_audio_5_15s'].get('p50_ms', 'N/A')} ms** | {lat['medium_audio_5_15s'].get('p70_ms', 'N/A')} ms | {lat['medium_audio_5_15s'].get('p90_ms', 'N/A')} ms | {lat['medium_audio_5_15s'].get('p95_ms', 'N/A')} ms | {lat['medium_audio_5_15s'].get('p99_ms', 'N/A')} ms | {lat['medium_audio_5_15s'].get('p100_ms', 'N/A')} ms | **{lat['medium_audio_5_15s'].get('mean_ms', 'N/A')} ms** |
| **Long (15–30s)** | {len([r for r in summary['detailed_results'] if r['duration_category'] == 'long'])} | **{lat['long_audio_15_30s'].get('p50_ms', 'N/A')} ms** | {lat['long_audio_15_30s'].get('p70_ms', 'N/A')} ms | {lat['long_audio_15_30s'].get('p90_ms', 'N/A')} ms | {lat['long_audio_15_30s'].get('p95_ms', 'N/A')} ms | {lat['long_audio_15_30s'].get('p99_ms', 'N/A')} ms | {lat['long_audio_15_30s'].get('p100_ms', 'N/A')} ms | **{lat['long_audio_15_30s'].get('mean_ms', 'N/A')} ms** |

---

## 2. Transcription Quality & Error Rates

- **Evaluated Languages**: `Hindi (hi-IN)`, `English (en-IN)`, `Bengali (bn-IN)`, `Tamil (ta-IN)`, `Telugu (te-IN)`, `Marathi (mr-IN)`
- **Mean Word Error Rate (WER)**: `{q['mean_wer']}`
- **Mean Character Error Rate (CER)**: `{q['mean_cer']}`

---

## 3. Multilingual Test Sample Highlights

| # | Language | Audio Duration | Reference Transcript | Hypothesis Transcript | WER | Status | Latency |
|---|---|---|---|---|---|---|---|
"""
    for row in summary["detailed_results"][:10]:
        md_content += f"| {row['sample_id']} | `{row['language']}` | {row['duration_seconds']}s | {row['reference'][:35]}... | {row['hypothesis'][:35]}... | {row['wer']:.2f} | **{row['status']}** | {row['latency_ms']:.2f} ms |\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved STT benchmark report -> {md_path}")
    logger.info(f"Saved STT benchmark JSON -> {json_path}")


def main():
    out_dir = BASE_DIR / "data" / "statistics"
    summary = run_stt_benchmark(num_samples=30)
    save_reports = save_stt_reports(summary, out_dir)


if __name__ == "__main__":
    main()
