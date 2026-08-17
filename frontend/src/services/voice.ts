const API_BASE_URL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL.replace(/\/+$/, '')}/api/v1`
  : '/api/v1';

export interface Citation {
  chunk_id: string;
  source_passage_id?: string;
  relevance_score?: number;
  snippet?: string;
}

export interface CandidateChunk {
  chunk_id: string;
  passage_id?: string;
  text: string;
  dense_score?: number;
  bm25_score?: number;
  rrf_score?: number;
  reranker_score?: number;
  is_selected?: boolean | number;
}

export interface RAGLatency {
  stt_ms?: number;
  normalization_ms?: number;
  analysis_ms?: number;
  guardrail_pre_ms?: number;
  dense_retrieval_ms?: number;
  bm25_retrieval_ms?: number;
  reranking_ms?: number;
  retrieval_total_ms?: number;
  context_prep_ms?: number;
  generation_ms?: number;
  verification_ms?: number;
  total_ms: number;
}

export interface RAGResult {
  transcript?: string;
  stt_language?: string;
  stt_provider?: string;
  query: string;
  normalized_query: string;
  detected_language: string;
  strategy: string;
  answer: string;
  grounded: boolean;
  confidence: number;
  citations: Citation[];
  abstained: boolean;
  abstention_reason?: string | null;
  retrieved_chunks: CandidateChunk[];
  latency: RAGLatency;
  request_id?: string;
}

export interface TranscribeResult {
  transcript: string;
  language_code: string;
  provider: string;
  model: string;
  duration_ms: number;
  latency: Record<string, number>;
  request_id: string;
}

export async function transcribeAudio(
  audioBlob: Blob,
  language?: string,
  model?: string,
): Promise<TranscribeResult> {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');
  if (language && language !== 'auto') {
    formData.append('language', language);
  }
  if (model) {
    formData.append('model', model);
  }

  const res = await fetch(`${API_BASE_URL}/voice/transcribe`, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();
  if (!res.ok || !json.success) {
    throw new Error(json?.error?.message || json?.message || 'Transcription failed');
  }
  return json.data;
}

export async function queryVoiceRAG(
  audioBlob: Blob,
  strategy = 'adaptive',
  language?: string,
  topK = 5,
): Promise<RAGResult> {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');
  formData.append('strategy', strategy);
  if (language && language !== 'auto') {
    formData.append('language', language);
  }
  formData.append('top_k', topK.toString());

  const res = await fetch(`${API_BASE_URL}/voice/query`, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();
  if (!res.ok || !json.success) {
    throw new Error(json?.error?.message || json?.message || 'Voice RAG query failed');
  }
  return json.data;
}

export async function queryTextRAG(
  query: string,
  strategy = 'adaptive',
  topK = 5,
): Promise<RAGResult> {
  const res = await fetch(`${API_BASE_URL}/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      strategy,
      top_k: topK,
      enable_reranking: true,
    }),
  });

  const json = await res.json();
  if (!res.ok || !json.success) {
    throw new Error(json?.error?.message || json?.message || 'Text RAG query failed');
  }
  return json.data;
}

export async function inspectRetrieval(
  query: string,
  strategy = 'adaptive',
  topK = 5,
  enableReranking = true,
) {
  const res = await fetch(`${API_BASE_URL}/rag/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      strategy,
      top_k: topK,
      enable_reranking: enableReranking,
    }),
  });

  const json = await res.json();
  if (!res.ok || !json.success) {
    throw new Error(json?.error?.message || json?.message || 'Inspection failed');
  }
  return json.data;
}

export async function getRAGInfo() {
  const res = await fetch(`${API_BASE_URL}/rag/info`);
  const json = await res.json();
  if (!res.ok || !json.success) {
    throw new Error('Failed to fetch RAG provider info');
  }
  return json.data;
}
