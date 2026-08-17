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
  p50: 24.897,
  p70: 31.01,
  p90: 46.48,
  p95: 51.042,
  p99: 56.753,
  p100: 59.955,
  mean: 23.816,
  target: 200.0,
  compliance: 'PASS_100_PERCENT',
  totalQueries: 141,
  dataset: 'MSMARCO-XI Hindi (99,925 passages)',
};

export const STAGE_LATENCY_BREAKDOWN: StageLatency[] = [
  {
    stage: 'Normalization & Analysis',
    shortLabel: 'Query Prep',
    p50: 0.12,
    p95: 0.18,
    p100: 0.22,
    mean: 0.15,
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
    p50: 4.83,
    p95: 31.59,
    p100: 35.75,
    mean: 12.79,
    description: 'Multilingual-E5-Small 384-d cosine inner product search',
    color: '#3b82f6', // blue
  },
  {
    stage: 'Sparse BM25 Retrieval',
    shortLabel: 'Okapi BM25',
    p50: 0.15,
    p95: 31.34,
    p100: 35.52,
    mean: 10.88,
    description: 'RankBM25 inverted subword lexical index score matching',
    color: '#6366f1', // indigo
  },
  {
    stage: 'RRF Fusion & Deduplication',
    shortLabel: 'RRF Fusion',
    p50: 0.18,
    p95: 0.24,
    p100: 0.31,
    mean: 0.2,
    description: 'Reciprocal Rank Fusion (K=60) with document-level deduplication',
    color: '#8b5cf6', // purple
  },
  {
    stage: 'Cross-Encoder Reranker',
    shortLabel: 'Reranker',
    p50: 4.17,
    p95: 5.61,
    p100: 11.25,
    mean: 4.39,
    description: 'mmarco-mMiniLMv2-L12-H384-v1 joint Transformer token cross-attention',
    color: '#f59e0b', // amber
  },
  {
    stage: 'Context Budgeting & Prep',
    shortLabel: 'Context Prep',
    p50: 0.01,
    p95: 0.02,
    p100: 0.02,
    mean: 0.01,
    description: 'Top-5 chunk allocation, token budget enforcement (<8k chars)',
    color: '#14b8a6', // teal
  },
  {
    stage: 'Grounded Generation',
    shortLabel: 'LLM Synthesis',
    p50: 0.0,
    p95: 20.61,
    p100: 20.72,
    mean: 5.66,
    description: 'Strict factual response generation conditioned exclusively on retrieved chunks',
    color: '#a855f7', // violet
  },
  {
    stage: 'Grounding Verification',
    shortLabel: 'Claim Proof',
    p50: 0.01,
    p95: 0.07,
    p100: 0.1,
    mean: 0.02,
    description: 'N-gram claim verification and citation pointer validation',
    color: '#ec4899', // pink
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
