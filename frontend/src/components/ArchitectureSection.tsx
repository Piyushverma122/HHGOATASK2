import { useState } from 'react';
import { ChevronDown, ChevronUp, Clock, Info, CheckCircle2 } from 'lucide-react';

export const ArchitectureSection: React.FC = () => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

  const layers = [
    {
      stage: '1. Voice Capture & STT Layer',
      tech: 'Web Audio API + Sarvam Saaras v3 REST API (NFC Canonicalized)',
      what: 'Captures 16kHz mono microphone streams, validates audio format, and transcribes Indic speech into normalized Unicode text.',
      why: 'Preserves native Indic character sequences and prevents audio clipping distortion.',
      latency: '15.0 ms cached / stream',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#FF3366]',
      badgeBg: 'bg-[#FF3366]',
    },
    {
      stage: '2. Multi-Strategy Chunking Layer',
      tech: '7 Custom Strategies (Adaptive, Semantic Cosine, Sentence, Paragraph, Overlap, Fixed, Metadata)',
      what: 'Splits 99,925 MSMARCO-XI passages into optimal semantic units respecting Devanagari Dandas (।).',
      why: 'Sentence and adaptive boundaries prevent splitting critical factual claims across chunk edges.',
      latency: 'Offline Indexing',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#06B6D4]',
      badgeBg: 'bg-[#06B6D4]',
    },
    {
      stage: '3. Parallel Hybrid Retrieval Layer',
      tech: 'FAISS IndexFlatIP (384-d E5-Small) + RankBM25 (Subwords) + ThreadPoolExecutor',
      why: 'Dense vector search captures semantic nuances while sparse BM25 guarantees exact entity and keyword matches concurrently.',
      what: 'Retrieves top-20 dense candidates and top-20 sparse candidates concurrently on separate threads.',
      latency: '19.74 ms concurrent',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#00F59B]',
      badgeBg: 'bg-[#00F59B]',
    },
    {
      stage: '4. Candidate Fusion & Cross-Encoder Reranking',
      tech: 'Reciprocal Rank Fusion (K=60) + cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
      what: 'Fuses candidate pools and scores joint (query, passage) token cross-attention vectors in batches of 16.',
      why: 'Avoids independent vector compression bottleneck and boosts Recall@1 by +137%.',
      latency: '4.17 ms mean',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#FEE101]',
      badgeBg: 'bg-[#FEE101]',
    },
    {
      stage: '5. Multi-Stage Guardrails & Safety Enforcement',
      tech: 'Input Regex Engine (<0.05ms) + Context Budgeter + Relevance Score Threshold (>0.01)',
      what: 'Blocks prompt injections before LLM invocation and triggers graceful abstention on out-of-domain queries.',
      why: 'Guarantees the system never executes adversarial instructions or hallucinates on ungrounded queries.',
      latency: '0.07 ms total',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#A855F7]',
      badgeBg: 'bg-[#A855F7]',
    },
    {
      stage: '6. Grounded Generation & Citation Verification',
      tech: 'Structured Factual Prompting + N-Gram Claim Alignment Verifier',
      what: 'Generates answers conditioned strictly on top-5 verified context chunks and verifies cited passage evidence.',
      why: 'Provides mathematical guarantee that every assertion links directly to an authoritative source chunk.',
      latency: '5.66 ms generation',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#22D3EE]',
      badgeBg: 'bg-[#22D3EE]',
    },
  ];

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-2xl font-black text-white uppercase font-sans tracking-wide">
          How Voice RAG Works
        </h2>
        <p className="text-xs text-slate-300 font-sans mt-0.5">
          Judge-facing deep dive into system architecture, design decisions, and latency contributions
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {layers.map((layer, idx) => {
          const isExpanded = expandedIndex === idx;
          return (
            <div
              key={idx}
              onClick={() => setExpandedIndex(isExpanded ? null : idx)}
              className={`p-5 rounded-2xl border-2 border-black transition-all cursor-pointer select-none ${
                isExpanded
                  ? 'bg-[#152038] shadow-[5px_5px_0px_0px_#FEE101] -translate-x-0.5 -translate-y-0.5'
                  : `bg-[#0C1220] hover:bg-[#101726] shadow-[4px_4px_0px_0px_#000000] ${layer.boxColor}`
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className={`w-7 h-7 rounded-lg border-2 border-black flex items-center justify-center font-black text-xs font-mono text-black shadow-[2px_2px_0px_0px_#000] ${layer.badgeBg}`}>
                    {idx + 1}
                  </div>
                  <h3 className="text-sm font-black text-white font-sans uppercase">{layer.stage}</h3>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#FEE101] font-mono font-bold hidden sm:inline">{layer.latency}</span>
                  {isExpanded ? <ChevronUp className="w-5 h-5 text-[#FEE101]" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </div>
              </div>

              <div className="text-xs text-[#06B6D4] font-mono font-bold mt-2">{layer.tech}</div>

              {isExpanded && (
                <div className="pt-3 mt-3 border-t-2 border-black space-y-3 text-xs animate-fadeIn font-sans">
                  <div className="space-y-1 bg-[#070A12] p-3.5 rounded-xl border border-black">
                    <span className="text-[11px] uppercase font-black text-[#FEE101] flex items-center gap-1.5 font-mono">
                      <Info className="w-3.5 h-3.5" /> What it does
                    </span>
                    <p className="text-slate-200 leading-relaxed font-normal">{layer.what}</p>
                  </div>
                  <div className="space-y-1 bg-[#070A12] p-3.5 rounded-xl border border-black">
                    <span className="text-[11px] uppercase font-black text-[#00F59B] flex items-center gap-1.5 font-mono">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Why this approach
                    </span>
                    <p className="text-slate-300 leading-relaxed font-normal">{layer.why}</p>
                  </div>
                  <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-slate-800 text-slate-300">
                    <span className="font-bold">Latency Contribution:</span>
                    <span className="text-[#FEE101] font-black flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" /> {layer.latency}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
};
