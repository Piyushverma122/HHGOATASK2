import React from 'react';
import { Database } from 'lucide-react';
import type { ApiStatus } from '../services/api';
import type { Route } from './TopNav';
import { Hero } from './Hero';
import { PipelineVisual } from './PipelineVisual';
import { PerformanceHeroCard } from './PerformanceHeroCard';
import { TechnicalCapabilityGrid } from './TechnicalCapabilityGrid';
import { MultilingualSection } from './MultilingualSection';
import { ArchitectureSection } from './ArchitectureSection';
import { RequirementChecklist } from './RequirementChecklist';

interface DashboardViewProps {
  apiStatus: ApiStatus;
  onNavigate: (route: Route) => void;
  onRefresh: () => void;
  checking: boolean;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-12 py-4 relative">
      {/* 1. Product Hero */}
      <Hero onNavigate={onNavigate} />

      {/* 2. Interactive Pipeline Visualization */}
      <PipelineVisual />

      {/* 3. Performance Hero Benchmark Card (<200ms Compliance) */}
      <PerformanceHeroCard onNavigate={onNavigate} />

      {/* 4. Technical Capability Grid (6 Cards) */}
      <TechnicalCapabilityGrid />

      {/* 5. 7-Language Indic Multilingual Section */}
      <MultilingualSection onSelectFixture={() => onNavigate('/voice')} />

      {/* 6. Dataset Provenance Card (Neo-Brutalist) */}
      <section className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 shadow-[6px_6px_0px_0px_#000000] space-y-4">
        <div className="flex items-center justify-between border-b-2 border-black pb-3">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-[#06B6D4]" />
            <h2 className="text-lg font-black text-white uppercase font-sans">
              MSMARCO-XI Dataset Provenance
            </h2>
          </div>
          <span className="px-3 py-1 rounded-full text-xs font-black bg-[#06B6D4] text-black border-2 border-black font-mono shadow-[2px_2px_0px_0px_#000]">
            PARQUET PERSISTED
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#FEE101]">
            <div className="text-xs uppercase font-black text-slate-300 font-mono">Total Passages</div>
            <div className="text-3xl font-mono font-black text-[#FEE101] mt-1">99,925</div>
          </div>
          <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#06B6D4]">
            <div className="text-xs uppercase font-black text-slate-300 font-mono">Validation Queries</div>
            <div className="text-3xl font-mono font-black text-[#06B6D4] mt-1">9,994</div>
          </div>
          <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#00F59B]">
            <div className="text-xs uppercase font-black text-slate-300 font-mono">Supported Languages</div>
            <div className="text-3xl font-mono font-black text-[#00F59B] mt-1">14 (7 Demo)</div>
          </div>
          <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 shadow-[3px_3px_0px_0px_#FF0080]">
            <div className="text-xs uppercase font-black text-slate-300 font-mono">Vector Embeddings</div>
            <div className="text-3xl font-mono font-black text-[#FF0080] mt-1">384-dim E5</div>
          </div>
        </div>
      </section>

      {/* 7. Architecture Deep Dive ("How it Works") */}
      <ArchitectureSection />

      {/* 8. Judge Requirements Coverage Audit */}
      <RequirementChecklist />
    </div>
  );
};
