import { useState } from 'react';
import {
  Search,
  ArrowLeft,
  Layers,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { inspectRetrieval } from '../services/voice';

interface RetrievalInspectorViewProps {
  onBack: () => void;
}

export function RetrievalInspectorView({ onBack }: RetrievalInspectorViewProps) {
  const [query, setQuery] = useState('भारत की राजधानी क्या है और यह कहाँ स्थित है?');
  const [strategy, setStrategy] = useState('adaptive');
  const [topK, setTopK] = useState(5);
  const [enableRerank, setEnableRerank] = useState(true);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'final' | 'dense' | 'bm25' | 'fused' | 'reranked'>('final');
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  const presets = [
    { lang: 'Hindi', q: 'भारत की राजधानी क्या है और यह कहाँ स्थित है?' },
    { lang: 'English', q: 'What is the capital of India and where is the central government located?' },
    { lang: 'Hinglish', q: 'India ki capital kya hai aur ye kaha par situated hai?' },
    { lang: 'Bengali', q: 'ভারতের রাজধানী কী এবং এটি কোথায় অবস্থিত?' },
    { lang: 'Tamil', q: 'இந்தியாவின் தலைநகரம் எது மற்றும் அரசு எங்கு அமைந்துள்ளது?' },
    { lang: 'Telugu', q: 'భారతదేశ రాజధాని ఏది మరియు ప్రభుత్వం ఎక్కడ ఉంది?' },
    { lang: 'Marathi', q: 'भारताची राजधानी कोणती आहे आणि सरकार कुठे स्थित आहे?' },
  ];

  const handleInspect = async (searchQuery = query) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await inspectRetrieval(searchQuery, strategy, topK, enableRerank);
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Retrieval inspection failed');
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (chunkId: string) => {
    setExpandedChunkId((prev) => (prev === chunkId ? null : chunkId));
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <button
            onClick={onBack}
            className="neo-btn inline-flex items-center gap-2 text-xs font-bold text-black px-3 py-1.5 rounded-lg bg-[#FEE101] transition mb-2 uppercase font-sans cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </button>
          <div className="flex items-center gap-3">
            <h2 className="text-3xl sm:text-4xl font-serif-claude font-bold text-white">
              Hybrid Retrieval Inspector
            </h2>
            <span className="px-3 py-1 rounded-full text-xs font-black bg-[#06B6D4] text-black border-2 border-black shadow-[2px_2px_0px_0px_#000] flex items-center gap-1.5 font-mono uppercase">
              <Layers className="w-3.5 h-3.5" />
              FAISS + BM25 + RRF + Cross-Encoder
            </span>
          </div>
          <p className="text-sm text-slate-300 font-sans mt-1">
            Inspect intermediate candidate pools, rank fusion distribution, and joint Transformer attention scores
          </p>
        </div>
      </div>

      {/* Query Bar & Controls */}
      <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 space-y-5 shadow-[6px_6px_0px_0px_#000000]">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleInspect()}
              placeholder="Enter search query in Hindi, English, Hinglish, Bengali, Tamil, Telugu, Marathi..."
              className="w-full pl-10 pr-4 py-3 rounded-xl bg-[#070A12] border-2 border-black text-sm text-white focus:outline-none focus:border-[#FEE101] placeholder:text-slate-500 font-sans"
            />
          </div>

          <div className="flex items-center gap-2">
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="bg-[#070A12] border-2 border-black text-white text-xs font-bold font-sans rounded-xl px-3 py-3 focus:outline-none focus:border-[#FEE101] cursor-pointer shadow-[2px_2px_0px_0px_#000]"
            >
              <option value="adaptive">⚡ Adaptive (Auto)</option>
              <option value="semantic">🧠 Semantic Cosine</option>
              <option value="sentence">🎯 Sentence-Aware</option>
              <option value="paragraph">📑 Paragraph-Aware</option>
              <option value="overlap">🔄 Overlap Chunking</option>
              <option value="fixed">📏 Fixed-Size</option>
              <option value="metadata">🏷️ Metadata-Informed</option>
            </select>

            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-[#070A12] border-2 border-black text-white text-xs font-bold font-mono rounded-xl px-3 py-3 focus:outline-none focus:border-[#FEE101] cursor-pointer shadow-[2px_2px_0px_0px_#000]"
            >
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={8}>Top 8</option>
              <option value={10}>Top 10</option>
            </select>

            <label className="flex items-center gap-1.5 px-3 py-3 rounded-xl bg-[#070A12] border-2 border-black text-xs font-bold text-slate-200 cursor-pointer select-none font-sans shadow-[2px_2px_0px_0px_#000]">
              <input
                type="checkbox"
                checked={enableRerank}
                onChange={(e) => setEnableRerank(e.target.checked)}
                className="w-3.5 h-3.5 accent-[#FEE101] rounded"
              />
              <span>Reranker</span>
            </label>

            <button
              onClick={() => handleInspect()}
              disabled={loading}
              className="neo-btn-cyan px-5 py-3 rounded-xl text-xs font-black uppercase font-sans flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <span>Inspecting...</span>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Inspect Pipeline</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* 1-Click Preset Queries */}
        <div className="space-y-1.5">
          <span className="text-[10px] uppercase font-black tracking-wider text-slate-400 font-mono">
            1-Click Multilingual Demo Queries:
          </span>
          <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
            {presets.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuery(p.q);
                  handleInspect(p.q);
                }}
                className="px-3 py-1.5 rounded-lg bg-[#070A12] hover:bg-[#121826] border-2 border-black text-[11px] font-sans font-bold text-slate-200 hover:text-[#FEE101] whitespace-nowrap transition cursor-pointer shadow-[2px_2px_0px_0px_#000]"
              >
                <strong className="text-[#06B6D4] font-mono mr-1">{p.lang}:</strong> {p.q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-[#FF3366] text-white border-2 border-black rounded-xl p-4 text-xs font-bold font-sans shadow-[4px_4px_0px_0px_#000]">
          {error}
        </div>
      )}

      {/* Results View */}
      {data ? (
        <div className="space-y-6">
          {/* Summary Metric Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-[#0C1220] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#FEE101]">
              <span className="text-[10px] uppercase font-black text-slate-400 font-mono">Final Chunks</span>
              <div className="text-2xl font-black text-[#FEE101] font-mono mt-0.5">
                {data.final_chunks?.length || 0}
              </div>
            </div>
            <div className="bg-[#0C1220] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#06B6D4]">
              <span className="text-[10px] uppercase font-black text-slate-400 font-mono">Dense Candidates</span>
              <div className="text-2xl font-black text-[#06B6D4] font-mono mt-0.5">
                {data.dense_candidates?.length || 0}
              </div>
            </div>
            <div className="bg-[#0C1220] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#00F59B]">
              <span className="text-[10px] uppercase font-black text-slate-400 font-mono">BM25 Candidates</span>
              <div className="text-2xl font-black text-[#00F59B] font-mono mt-0.5">
                {data.bm25_candidates?.length || 0}
              </div>
            </div>
            <div className="bg-[#0C1220] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#FF0080]">
              <span className="text-[10px] uppercase font-black text-slate-400 font-mono">Reranker</span>
              <div className="text-2xl font-black text-[#FF0080] font-mono mt-0.5">
                {enableRerank ? 'ACTIVE' : 'OFF'}
              </div>
            </div>
          </div>

          {/* Candidate Stage Tabs (Neo-Brutalist) */}
          <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 space-y-5 shadow-[6px_6px_0px_0px_#000000]">
            <div className="flex items-center justify-between border-b-2 border-black pb-4 flex-wrap gap-2">
              <div className="flex items-center gap-2 flex-wrap">
                {[
                  { key: 'final', label: 'Final Context', color: 'bg-[#FEE101] text-black' },
                  { key: 'reranked', label: 'Cross-Encoder Reranked', color: 'bg-[#FF0080] text-white' },
                  { key: 'fused', label: 'RRF Fusion', color: 'bg-[#A855F7] text-white' },
                  { key: 'dense', label: 'Dense FAISS', color: 'bg-[#06B6D4] text-black' },
                  { key: 'bm25', label: 'Sparse BM25', color: 'bg-[#00F59B] text-black' },
                ].map((tab) => {
                  const isActive = activeTab === tab.key;
                  return (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key as any)}
                      className={`px-4 py-2 rounded-xl text-xs font-bold font-sans uppercase tracking-wider transition-all cursor-pointer border-2 border-black ${
                        isActive
                          ? `${tab.color} shadow-[3px_3px_0px_0px_#000000] -translate-x-0.5 -translate-y-0.5`
                          : 'bg-[#070A12] text-slate-300 hover:text-white shadow-[2px_2px_0px_0px_#000]'
                      }`}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Candidate List */}
            <div className="space-y-3">
              {(() => {
                let candidates = [];
                if (activeTab === 'final') candidates = data.final_chunks || [];
                else if (activeTab === 'reranked') candidates = data.reranked_candidates || [];
                else if (activeTab === 'fused') candidates = data.fused_candidates || [];
                else if (activeTab === 'dense') candidates = data.dense_candidates || [];
                else if (activeTab === 'bm25') candidates = data.bm25_candidates || [];

                if (candidates.length === 0) {
                  return (
                    <div className="text-center py-10 text-xs text-slate-400 font-sans italic">
                      No candidate passages in this stage.
                    </div>
                  );
                }

                return candidates.map((c: any, idx: number) => {
                  const chunkId = c.chunk_id || `candidate-${idx}`;
                  const isExp = expandedChunkId === chunkId;

                  return (
                    <div
                      key={chunkId}
                      className={`rounded-xl border-2 border-black transition-all ${
                        isExp
                          ? 'bg-[#152038] shadow-[4px_4px_0px_0px_#FEE101]'
                          : 'bg-[#070A12] hover:bg-[#101726] shadow-[3px_3px_0px_0px_#000000]'
                      }`}
                    >
                      <div
                        onClick={() => toggleExpand(chunkId)}
                        className="p-4 flex items-center justify-between gap-3 cursor-pointer select-none"
                      >
                        <div className="flex items-center gap-3">
                          <span className="w-7 h-7 rounded bg-[#FEE101] text-black border border-black flex items-center justify-center font-mono font-black text-xs shadow-[1px_1px_0px_0px_#000]">
                            #{idx + 1}
                          </span>
                          <span className="font-mono text-xs font-bold text-white">
                            {chunkId}
                          </span>
                        </div>

                        <div className="flex items-center gap-3">
                          {c.rerank_score !== undefined && (
                            <span className="px-2.5 py-0.5 rounded bg-[#FF0080] text-white border border-black text-[11px] font-mono font-bold shadow-[1px_1px_0px_0px_#000]">
                              Rerank: {c.rerank_score.toFixed(4)}
                            </span>
                          )}
                          {c.dense_score !== undefined && (
                            <span className="px-2.5 py-0.5 rounded bg-[#06B6D4] text-black border border-black text-[11px] font-mono font-bold shadow-[1px_1px_0px_0px_#000]">
                              Dense: {c.dense_score.toFixed(4)}
                            </span>
                          )}
                          {c.bm25_score !== undefined && (
                            <span className="px-2.5 py-0.5 rounded bg-[#00F59B] text-black border border-black text-[11px] font-mono font-bold shadow-[1px_1px_0px_0px_#000]">
                              BM25: {c.bm25_score.toFixed(2)}
                            </span>
                          )}
                          {isExp ? <ChevronUp className="w-4 h-4 text-[#FEE101]" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                        </div>
                      </div>

                      {isExp && (
                        <div className="px-4 pb-4 pt-1 space-y-3 text-xs border-t-2 border-black animate-fadeIn">
                          <div className="space-y-1">
                            <span className="text-[10px] uppercase font-black text-[#FEE101] font-mono">
                              Passage Snippet
                            </span>
                            <p className="text-slate-100 font-serif-claude text-sm sm:text-base leading-relaxed bg-[#070A12] p-4 rounded-xl border-2 border-black shadow-[2px_2px_0px_0px_#000] italic">
                              "{c.text || c.content || c.snippet || 'Passage content...'}"
                            </p>
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px] font-mono text-slate-300 border-t border-slate-800">
                            <div>
                              <span className="text-slate-500 block text-[10px] uppercase font-bold">PASSAGE ID</span>
                              <span className="text-white font-bold">{c.passage_id || 'N/A'}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 block text-[10px] uppercase font-bold">CHARS</span>
                              <span className="text-[#06B6D4] font-bold">{(c.text || c.content || '').length}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 block text-[10px] uppercase font-bold">RRF RANK</span>
                              <span className="text-[#00F59B] font-bold">{c.rrf_rank || idx + 1}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 block text-[10px] uppercase font-bold">STRATEGY</span>
                              <span className="text-[#FEE101] font-bold">{strategy}</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                });
              })()}
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-12 text-center space-y-4 shadow-[6px_6px_0px_0px_#000]">
          <div className="w-14 h-14 rounded-2xl bg-[#06B6D4] text-black border-2 border-black flex items-center justify-center mx-auto shadow-[3px_3px_0px_0px_#000]">
            <Search className="w-7 h-7" />
          </div>
          <h3 className="text-lg font-black text-white uppercase font-sans">Inspect Real Hybrid Retrieval</h3>
          <p className="text-xs text-slate-300 max-w-md mx-auto font-sans">
            Select a strategy or demo query above and click 'Inspect Pipeline' to visualize intermediate candidate scores across Dense FAISS, BM25, RRF fusion, and Cross-Encoder layers.
          </p>
          <button
            onClick={() => handleInspect()}
            className="neo-btn-yellow px-6 py-2.5 rounded-xl text-xs font-black uppercase font-sans cursor-pointer"
          >
            Run Default Hindi Query
          </button>
        </div>
      )}
    </div>
  );
}
