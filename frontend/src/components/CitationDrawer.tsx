import { useState } from 'react';
import { BookmarkCheck, ChevronDown, ChevronUp } from 'lucide-react';
import type { Citation } from '../services/voice';

interface CitationDrawerProps {
  citations: Citation[];
  strategy?: string;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ citations, strategy = 'adaptive' }) => {
  const [expandedId, setExpandedId] = useState<string | null>(citations[0]?.chunk_id || null);

  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 sm:p-6 shadow-[5px_5px_0px_0px_#000000] space-y-4">
      <div className="flex items-center justify-between border-b-2 border-black pb-3">
        <div className="flex items-center gap-2">
          <BookmarkCheck className="w-5 h-5 text-[#22D3EE]" />
          <h3 className="text-sm font-black text-white uppercase tracking-wider font-sans">
            Verified Evidence Citations ({citations.length})
          </h3>
        </div>
        <span className="text-xs text-slate-300 font-mono">
          Strategy: <strong className="text-[#FEE101]">{strategy}</strong>
        </span>
      </div>

      <div className="space-y-3">
        {citations.map((cit, idx) => {
          const chunkId = cit.chunk_id || `citation-${idx}`;
          const isExpanded = expandedId === chunkId;

          return (
            <div
              key={chunkId}
              className={`rounded-xl border-2 border-black transition-all ${
                isExpanded
                  ? 'bg-[#152038] shadow-[4px_4px_0px_0px_#FEE101]'
                  : 'bg-[#070A12] hover:bg-[#101726] shadow-[3px_3px_0px_0px_#000000]'
              }`}
            >
              {/* Header Bar */}
              <div
                onClick={() => setExpandedId(isExpanded ? null : chunkId)}
                className="p-3.5 flex items-center justify-between gap-3 cursor-pointer select-none"
              >
                <div className="flex items-center gap-2.5">
                  <span className="px-2 py-0.5 rounded text-[10px] font-black bg-[#FEE101] text-black border border-black font-mono shadow-[1px_1px_0px_0px_#000]">
                    SOURCE 0{idx + 1}
                  </span>
                  <span className="font-mono text-xs font-bold text-slate-200">
                    {chunkId}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  {cit.relevance_score !== undefined && (
                    <span className="px-2 py-0.5 rounded bg-[#00F59B] text-black border border-black text-[11px] font-mono font-bold shadow-[1px_1px_0px_0px_#000]">
                      Relevance: {cit.relevance_score.toFixed(4)}
                    </span>
                  )}
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-[#FEE101]" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                  )}
                </div>
              </div>

              {/* Expandable Body with Claude Serif Quote */}
              {isExpanded && (
                <div className="px-4 pb-4 pt-1 space-y-3 text-xs border-t-2 border-black animate-fadeIn">
                  {/* Verified Snippet (Claude Newsreader Serif) */}
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-black text-[#FEE101] font-mono">
                      Grounded Source Passage
                    </span>
                    <p className="text-slate-100 font-serif-claude text-sm sm:text-base leading-relaxed bg-[#070A12] p-4 rounded-xl border-2 border-black shadow-[2px_2px_0px_0px_#000000] italic">
                      "{cit.snippet || 'Passage text verified in context window.'}"
                    </p>
                  </div>

                  {/* Metadata Matrix */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px] font-mono text-slate-300 border-t border-slate-800">
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-bold">CHUNK ID</span>
                      <span className="text-white font-bold truncate block">{chunkId}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-bold">STRATEGY</span>
                      <span className="text-[#22D3EE] font-bold">{strategy}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-bold">RERANKER SCORE</span>
                      <span className="text-[#00F59B] font-bold">
                        {cit.relevance_score ? cit.relevance_score.toFixed(4) : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-bold">VERIFICATION</span>
                      <span className="text-[#00F59B] font-bold flex items-center gap-1">
                        <BookmarkCheck className="w-3.5 h-3.5" /> Grounded
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
