import React from 'react';
import { Mic, Search, Sparkles } from 'lucide-react';
import type { Route } from './TopNav';

interface HeroProps {
  onNavigate: (route: Route) => void;
}

export const Hero: React.FC<HeroProps> = ({ onNavigate }) => {
  return (
    <section className="relative text-center space-y-6 max-w-4xl mx-auto py-8 sm:py-12">
      {/* Background Decorative Asset Watermark */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl h-80 opacity-15 pointer-events-none -z-10 flex items-center justify-center">
        <img
          src="/Assets/Sun%20rise.png"
          alt="Sun rise backdrop"
          className="w-full h-full object-contain filter invert contrast-200"
          onError={(e) => {
            (e.target as HTMLElement).style.display = 'none';
          }}
        />
      </div>

      {/* Pill Badge (Neo-Brutalist) */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#FEE101] text-black border-2 border-black font-mono text-xs font-black uppercase tracking-wider shadow-[3px_3px_0px_0px_#000000]">
        <Sparkles className="w-3.5 h-3.5" />
        <span>HH GOA 2026 • TASK 2</span>
      </div>

      {/* Hero Title (Claude Newsreader Editorial Serif) */}
      <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-serif-claude tracking-tight text-white leading-[1.08] font-bold">
        Voice-Enabled <br className="hidden sm:inline" />
        <span className="text-[#FEE101] italic underline decoration-[#FF0080] decoration-4 underline-offset-8">
          Retrieval-Augmented
        </span>{' '}
        <span className="text-[#22D3EE]">Generation</span>
      </h1>

      {/* Subtitle */}
      <p className="text-slate-300 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed font-sans font-medium">
        A multilingual voice-first RAG system that transforms speech into grounded answers using hybrid retrieval, reranking, guardrails and citation-aware generation.
      </p>

      {/* Hero Action CTAs (Neo-Brutalist High-Contrast Buttons) */}
      <div className="flex flex-wrap items-center justify-center gap-4 pt-3">
        <button
          onClick={() => onNavigate('/voice')}
          className="neo-btn-yellow px-8 py-3.5 rounded-xl font-sans font-black text-xs sm:text-sm flex items-center gap-2 cursor-pointer uppercase tracking-wider"
        >
          <Mic className="w-4 h-4" />
          <span>Launch Voice Studio</span>
        </button>
        <button
          onClick={() => onNavigate('/retrieval')}
          className="neo-btn-cyan px-8 py-3.5 rounded-xl font-sans font-black text-xs sm:text-sm flex items-center gap-2 cursor-pointer uppercase tracking-wider"
        >
          <Search className="w-4 h-4" />
          <span>Explore Retrieval</span>
        </button>
      </div>
    </section>
  );
};
