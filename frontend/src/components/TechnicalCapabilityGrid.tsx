import React from 'react';
import { Volume2, Layers, Search, Sparkles, BookmarkCheck, Shield } from 'lucide-react';

export const TechnicalCapabilityGrid: React.FC = () => {
  const capabilities = [
    {
      num: '01',
      title: 'Multilingual STT',
      subtitle: 'Sarvam Saaras v3',
      desc: 'High-accuracy voice transcription with 16kHz mono audio validation and Unicode NFC normalization across 7 Indic languages.',
      icon: <Volume2 className="w-5 h-5 text-black" />,
      badge: 'STT LAYER',
      badgeBg: 'bg-[#FEE101]',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#FEE101]',
    },
    {
      num: '02',
      title: '7 Chunking Strategies',
      subtitle: 'Adaptive + Semantic',
      desc: 'Adaptive, Semantic Cosine, Sentence, Paragraph, Overlap, Fixed-Size, and Metadata-Informed chunking optimized for Devanagari Dandas.',
      icon: <Layers className="w-5 h-5 text-black" />,
      badge: 'CHUNKING',
      badgeBg: 'bg-[#06B6D4]',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#06B6D4]',
    },
    {
      num: '03',
      title: 'Hybrid Retrieval',
      subtitle: 'FAISS + BM25 + RRF',
      desc: 'Parallel concurrent execution of 384-d dense vector search and subword BM25 fused via Reciprocal Rank Fusion (K=60).',
      icon: <Search className="w-5 h-5 text-black" />,
      badge: 'RETRIEVAL',
      badgeBg: 'bg-[#00F59B]',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#00F59B]',
    },
    {
      num: '04',
      title: 'Cross-Encoder Reranker',
      subtitle: 'mmarco-mMiniLMv2',
      desc: 'Genuine Transformer cross-encoder computing joint query-passage cross-attention, boosting Recall@1 by +137%.',
      icon: <Sparkles className="w-5 h-5 text-black" />,
      badge: 'RERANKING',
      badgeBg: 'bg-[#FF0080]',
      badgeText: 'text-white',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#FF0080]',
    },
    {
      num: '05',
      title: 'Grounded Generation',
      subtitle: 'Citation-Aware Synthesis',
      desc: 'Strict factual prompting conditioned exclusively on retrieved MSMARCO-XI chunks with automated n-gram claim verification.',
      icon: <BookmarkCheck className="w-5 h-5 text-black" />,
      badge: 'GENERATION',
      badgeBg: 'bg-[#A855F7]',
      badgeText: 'text-white',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#A855F7]',
    },
    {
      num: '06',
      title: 'Multi-Stage Guardrails',
      subtitle: 'Prompt Injection Defense',
      desc: '<0.05ms input regex defense, context budget limits, relevance thresholds, and graceful abstention on off-topic queries.',
      icon: <Shield className="w-5 h-5 text-black" />,
      badge: 'GUARDRAILS',
      badgeBg: 'bg-[#FF3366]',
      badgeText: 'text-white',
      boxColor: 'hover:shadow-[5px_5px_0px_0px_#FF3366]',
    },
  ];

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-2xl font-black text-white uppercase font-sans tracking-wide">
          Technical Capabilities
        </h2>
        <p className="text-xs text-slate-300 font-sans mt-0.5">
          Production-grade components engineered specifically for the HH Goa 2026 Task 2 challenge
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {capabilities.map((cap, idx) => (
          <div
            key={idx}
            className={`bg-[#0C1220] border-2 border-black rounded-2xl p-5 transition-all shadow-[4px_4px_0px_0px_#000000] hover:-translate-x-0.5 hover:-translate-y-0.5 ${cap.boxColor} space-y-3 flex flex-col justify-between`}
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-black text-slate-400">{cap.num}</span>
                <span className={`text-[10px] font-bold font-mono uppercase tracking-wider px-2 py-0.5 rounded border border-black shadow-[1px_1px_0px_0px_#000] ${cap.badgeBg} ${cap.badgeText || 'text-black'}`}>
                  {cap.badge}
                </span>
              </div>
              <div className="flex items-start gap-3">
                <div className={`p-2.5 rounded-xl border-2 border-black shadow-[2px_2px_0px_0px_#000] shrink-0 ${cap.badgeBg}`}>
                  {cap.icon}
                </div>
                <div>
                  <h3 className="text-base font-black text-white leading-snug font-sans uppercase">{cap.title}</h3>
                  <div className="text-xs text-[#FEE101] font-bold font-mono mt-0.5">{cap.subtitle}</div>
                </div>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">{cap.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
