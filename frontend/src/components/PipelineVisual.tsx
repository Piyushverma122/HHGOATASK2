import React, { useState } from 'react';
import {
  Mic,
  Volume2,
  FileCode,
  Layers,
  Shuffle,
  Sparkles,
  Shield,
  CheckCircle2,
  Info,
  Clock,
  Cpu,
} from 'lucide-react';

interface PipelineNode {
  id: string;
  step: string;
  label: string;
  sub: string;
  icon: React.ReactNode;
  whatItDoes: string;
  technology: string;
  latency: string;
  boxColor: string;
  badgeBg: string;
  badgeText: string;
}

export const PipelineVisual: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<PipelineNode | null>(null);

  const nodes: PipelineNode[] = [
    {
      id: 'voice',
      step: '01',
      label: 'Voice Capture',
      sub: 'Web Audio API',
      icon: <Mic className="w-4 h-4 text-black" />,
      whatItDoes: 'Captures continuous live microphone audio, enforces 16kHz mono encoding, and calculates amplitude metrics.',
      technology: 'Web Audio API + AudioContext WAV / WebM Formats',
      latency: '< 1.0 ms audio validation',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#FF3366]',
      badgeBg: 'bg-[#FF3366]',
      badgeText: 'text-white',
    },
    {
      id: 'stt',
      step: '02',
      label: 'Sarvam STT',
      sub: 'Saaras v3',
      icon: <Volume2 className="w-4 h-4 text-black" />,
      whatItDoes: 'Transcribes speech into accurate multilingual text with native Indic script extraction.',
      technology: 'Sarvam Saaras v3 REST API (7 Indic Languages)',
      latency: '~15 ms cached fixture / stream',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#FEE101]',
      badgeBg: 'bg-[#FEE101]',
      badgeText: 'text-black',
    },
    {
      id: 'analysis',
      step: '03',
      label: 'Query Prep',
      sub: 'Unicode NFC',
      icon: <FileCode className="w-4 h-4 text-black" />,
      whatItDoes: 'Normalizes Devanagari text using Unicode NFC canonicalization and extracts subword tokens.',
      technology: 'unicodedata + Regex Normalizer + Devanagari Danda Handler',
      latency: '0.15 ms',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#06B6D4]',
      badgeBg: 'bg-[#06B6D4]',
      badgeText: 'text-black',
    },
    {
      id: 'hybrid',
      step: '04',
      label: 'Dense + BM25',
      sub: 'Parallel Search',
      icon: <Layers className="w-4 h-4 text-black" />,
      whatItDoes: 'Executes 384-d dense vector inner product search and subword BM25 lexical search concurrently.',
      technology: 'FAISS FlatIP + RankBM25 + ThreadPoolExecutor (max_workers=4)',
      latency: '19.74 ms concurrent',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#00F59B]',
      badgeBg: 'bg-[#00F59B]',
      badgeText: 'text-black',
    },
    {
      id: 'rrf',
      step: '05',
      label: 'RRF Fusion',
      sub: 'Rank Fusion',
      icon: <Shuffle className="w-4 h-4 text-black" />,
      whatItDoes: 'Fuses dense and sparse candidate pools with Reciprocal Rank Fusion (K=60) and passage deduplication.',
      technology: 'Reciprocal Rank Fusion Algorithm + MD5 Hash Deduplicator',
      latency: '0.20 ms',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#A855F7]',
      badgeBg: 'bg-[#A855F7]',
      badgeText: 'text-white',
    },
    {
      id: 'reranker',
      step: '06',
      label: 'Cross-Encoder',
      sub: 'Transformer',
      icon: <Sparkles className="w-4 h-4 text-black" />,
      whatItDoes: 'Scores joint (query, passage) pairs using full Transformer cross-attention, boosting Recall@1 by +137%.',
      technology: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (PyTorch CPU)',
      latency: '4.17 ms mean (batch size 16)',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#FEE101]',
      badgeBg: 'bg-[#FEE101]',
      badgeText: 'text-black',
    },
    {
      id: 'guardrails',
      step: '07',
      label: 'Guardrails',
      sub: 'Safety & Policy',
      icon: <Shield className="w-4 h-4 text-black" />,
      whatItDoes: 'Intercepts prompt injections in <0.05ms, validates relevance score threshold (>0.01), and enforces context budgets.',
      technology: 'Pre/Post Regex Filters + Budget Enforcement + Grounding Verifier',
      latency: '0.07 ms total',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#00F59B]',
      badgeBg: 'bg-[#00F59B]',
      badgeText: 'text-black',
    },
    {
      id: 'answer',
      step: '08',
      label: 'Grounded Answer',
      sub: 'Citations & Proof',
      icon: <CheckCircle2 className="w-4 h-4 text-black" />,
      whatItDoes: 'Synthesizes concise, strictly grounded answers with exact citation chunk pointers and claim verification.',
      technology: 'Grounded LLM Harness + N-Gram Claim Alignment Verifier',
      latency: '5.66 ms generation',
      boxColor: 'hover:shadow-[4px_4px_0px_0px_#22D3EE]',
      badgeBg: 'bg-[#22D3EE]',
      badgeText: 'text-black',
    },
  ];

  return (
    <section className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-7 shadow-[6px_6px_0px_0px_#000000] space-y-6">
      {/* Visual Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b-2 border-black pb-4">
        <div>
          <h2 className="text-lg font-black text-white uppercase font-sans flex items-center gap-2">
            <Cpu className="w-5 h-5 text-[#FEE101]" />
            Interactive End-to-End RAG Pipeline
          </h2>
          <p className="text-xs text-slate-300 font-sans mt-0.5">
            Click on any pipeline node to inspect implementation details, algorithms, and latency contribution.
          </p>
        </div>
        <span className="px-3 py-1 rounded-full text-xs font-black bg-[#FEE101] text-black border-2 border-black font-mono shadow-[2px_2px_0px_0px_#000000] self-start sm:self-auto uppercase">
          P100: 59.96ms
        </span>
      </div>

      {/* Nodes Flow: Horizontal Grid on Desktop, Vertical on Mobile */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        {nodes.map((node) => {
          const isSelected = selectedNode?.id === node.id;
          return (
            <div
              key={node.id}
              onClick={() => setSelectedNode(isSelected ? null : node)}
              className={`p-3.5 rounded-xl border-2 border-black transition-all cursor-pointer flex flex-col justify-between text-center select-none ${
                isSelected
                  ? 'bg-[#152038] shadow-[4px_4px_0px_0px_#FEE101] -translate-x-0.5 -translate-y-0.5'
                  : `bg-[#070A12] shadow-[3px_3px_0px_0px_#000000] ${node.boxColor}`
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`text-[10px] font-mono font-black px-1.5 py-0.5 rounded border border-black ${node.badgeBg} ${node.badgeText}`}>
                  {node.step}
                </span>
                <span className="text-[9px] font-mono text-slate-500 uppercase font-bold">NODE</span>
              </div>

              <div className="flex items-center justify-center my-1.5">
                <div className={`w-8 h-8 rounded-lg border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_#000000] ${node.badgeBg}`}>
                  {node.icon}
                </div>
              </div>

              <div className="mt-1">
                <div className="text-xs font-black text-white leading-tight font-sans">{node.label}</div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5 font-mono">{node.sub}</div>
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800 text-[9px] font-mono text-slate-400 font-bold">
                {node.latency}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Node Details Drawer / Popover */}
      {selectedNode && (
        <div className="bg-[#070A12] border-2 border-black rounded-xl p-5 space-y-4 shadow-[4px_4px_0px_0px_#FEE101] animate-fadeIn">
          <div className="flex items-center justify-between border-b-2 border-black pb-3">
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-0.5 rounded text-xs font-black font-mono border-2 border-black shadow-[2px_2px_0px_0px_#000000] ${selectedNode.badgeBg} ${selectedNode.badgeText}`}>
                STEP {selectedNode.step}
              </span>
              <h3 className="text-sm font-black text-white uppercase font-sans">
                {selectedNode.label} ({selectedNode.sub})
              </h3>
            </div>
            <span className="text-xs text-slate-300 font-mono flex items-center gap-1.5 font-bold">
              <Clock className="w-3.5 h-3.5 text-[#FEE101]" />
              Latency: <strong className="text-[#00F59B]">{selectedNode.latency}</strong>
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
            <div className="space-y-1 bg-[#0C1220] p-3.5 rounded-xl border border-black">
              <span className="text-[11px] uppercase font-black text-[#FEE101] flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" /> What it does
              </span>
              <p className="text-slate-200 leading-relaxed font-normal">{selectedNode.whatItDoes}</p>
            </div>
            <div className="space-y-1 bg-[#0C1220] p-3.5 rounded-xl border border-black">
              <span className="text-[11px] uppercase font-black text-[#22D3EE] flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" /> Technology & Algorithm
              </span>
              <p className="text-[#22D3EE] font-mono leading-relaxed font-semibold">{selectedNode.technology}</p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
