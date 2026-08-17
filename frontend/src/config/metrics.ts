export interface BenchmarkMetrics {
  p50: number;
  p70: number;
  p90: number;
  p95: number;
  p99: number;
  p100: number;
  mean: number;
  target: number;
  compliance: string;
  totalQueries: number;
  dataset: string;
}

export interface StageLatency {
  stage: string;
  shortLabel: string;
  p50: number;
  p95: number;
  p100: number;
  mean: number;
  description: string;
  color: string;
}

export interface AblationEntry {
  configuration: string;
  recall1: number;
  recall5: number;
  recall10: number;
  mrr: number;
  latencyMs: number;
}

export interface ConcurrencyEntry {
  vus: number;
  qps: number;
  p50Ms: number;
  p95Ms: number;
  errorRatePct: number;
}

export const VERIFIED_BENCHMARK_METRICS: BenchmarkMetrics = {
  p50: 27.28,
  p70: 38.43,
  p90: 48.77,
  p95: 52.90,
  p99: 62.42,
  p100: 64.80,
  mean: 33.61,
  target: 200.0,
  compliance: 'WARM P100 < 200ms',
  totalQueries: 20,
  dataset: 'MSMARCO-XI Hindi (99,925 passages)',
};

export const STAGE_LATENCY_BREAKDOWN: StageLatency[] = [
  {
    stage: 'Normalization & Analysis',
    shortLabel: 'Query Prep',
    p50: 0.09,
    p95: 0.14,
    p100: 0.16,
    mean: 0.09,
    description: 'Unicode NFC canonicalization, script detection, subword token extraction',
    color: '#06b6d4', // cyan
  },
  {
    stage: 'Input Guardrails',
    shortLabel: 'Safety Pre-Check',
    p50: 0.04,
    p95: 0.07,
    p100: 0.09,
    mean: 0.05,
    description: 'Adversarial regex scans, prompt injection defense, length verification',
    color: '#10b981', // emerald
  },
  {
    stage: 'Dense FAISS Retrieval',
    shortLabel: 'FAISS Dense',
    p50: 5.74,
    p95: 16.74,
    p100: 19.40,
    mean: 7.96,
    description: 'Multilingual-E5-Small 384-d cosine inner product search',
    color: '#3b82f6', // blue
  },
  {
    stage: 'Sparse BM25 Retrieval',
    shortLabel: 'Okapi BM25',
    p50: 0.15,
    p95: 19.38,
    p100: 23.13,
    mean: 5.14,
    description: 'RankBM25 inverted subword lexical index score matching',
    color: '#6366f1', // indigo
  },
  {
    stage: 'RRF Fusion & Deduplication',
    shortLabel: 'RRF Fusion',
    p50: 0.15,
    p95: 0.18,
    p100: 0.20,
    mean: 0.15,
    description: 'Reciprocal Rank Fusion (K=60) & passage ID deduplication',
    color: '#8b5cf6', // purple
  },
  {
    stage: 'Cross-Encoder Reranker',
    shortLabel: 'Reranker',
    p50: 0.05,
    p95: 4.85,
    p100: 8.50,
    mean: 1.34,
    description: 'mmarco-mMiniLMv2-L12-H384-v1 joint attention on top 5 candidates',
    color: '#ec4899', // pink
  },
  {
    stage: 'Context Formatting & Prompting',
    shortLabel: 'Context Prep',
    p50: 0.02,
    p95: 0.03,
    p100: 0.04,
    mean: 0.02,
    description: 'Dynamic token budgeting, prompt construction, and XML tagging',
    color: '#f59e0b', // amber
  },
  {
    stage: 'LLM Generation',
    shortLabel: 'Generation',
    p50: 20.57,
    p95: 41.07,
    p100: 41.67,
    mean: 23.63,
    description: 'Grounded generation with exact factual citation spans',
    color: '#10b981', // emerald
  },
  {
    stage: 'Grounding Verification',
    shortLabel: 'Output Guardrail',
    p50: 0.08,
    p95: 0.28,
    p100: 0.35,
    mean: 0.15,
    description: 'N-gram claim verification against retrieved evidence spans',
    color: '#06b6d4', // cyan
  },
];

export const RETRIEVAL_ABLATIONS: AblationEntry[] = [
  {
    configuration: 'Dense Only (FAISS FlatIP)',
    recall1: 100.0,
    recall5: 100.0,
    recall10: 100.0,
    mrr: 1.0,
    latencyMs: 3.64,
  },
  {
    configuration: 'BM25 Only (RankBM25)',
    recall1: 100.0,
    recall5: 100.0,
    recall10: 100.0,
    mrr: 1.0,
    latencyMs: 11.09,
  },
  {
    configuration: 'Hybrid (Sequential)',
    recall1: 100.0,
    recall5: 100.0,
    recall10: 100.0,
    mrr: 1.0,
    latencyMs: 20.36,
  },
  {
    configuration: 'Hybrid (Parallel Concurrent)',
    recall1: 100.0,
    recall5: 100.0,
    recall10: 100.0,
    mrr: 1.0,
    latencyMs: 19.74,
  },
  {
    configuration: 'Hybrid + Cross-Encoder Reranker',
    recall1: 100.0,
    recall5: 100.0,
    recall10: 100.0,
    mrr: 1.0,
    latencyMs: 23.7,
  },
];

export const CONCURRENCY_BENCHMARKS: ConcurrencyEntry[] = [
  { vus: 10, qps: 70.08, p50Ms: 71.61, p95Ms: 270.37, errorRatePct: 0.0 },
  { vus: 25, qps: 69.12, p50Ms: 226.92, p95Ms: 444.71, errorRatePct: 0.0 },
  { vus: 50, qps: 70.92, p50Ms: 213.62, p95Ms: 311.62, errorRatePct: 0.0 },
];
