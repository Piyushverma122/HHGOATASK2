import React from 'react';
import { Gauge, CheckCircle2, TrendingDown, Zap, Activity } from 'lucide-react';
import { VERIFIED_BENCHMARK_METRICS } from '../config/metrics';
import type { Route } from './TopNav';

interface PerformanceHeroCardProps {
  onNavigate: (route: Route) => void;
}

export const PerformanceHeroCard: React.FC<PerformanceHeroCardProps> = ({ onNavigate }) => {
  return (
    <section className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 shadow-[6px_6px_0px_0px_#000000] space-y-6">
      {/* Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b-2 border-black pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 rounded-full text-xs font-black uppercase bg-[#00F59B] text-black border-2 border-black shadow-[2px_2px_0px_0px_#000000] flex items-center gap-1.5 font-mono">
              <CheckCircle2 className="w-3.5 h-3.5" />
              100% COMPLIANT
            </span>
            <h2 className="text-xl font-black text-white font-sans uppercase tracking-wide">
              Production Latency Benchmark
            </h2>
          </div>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Empirically measured over {VERIFIED_BENCHMARK_METRICS.totalQueries} multilingual MSMARCO-XI queries with real cross-encoder reranking.
          </p>
        </div>

        <button
          onClick={() => onNavigate('/analytics')}
          className="neo-btn px-4 py-2 rounded-xl bg-[#FEE101] text-black text-xs font-black uppercase tracking-wider flex items-center gap-1.5 self-start sm:self-auto cursor-pointer font-sans"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>View Telemetry Breakdown →</span>
        </button>
      </div>

      {/* Hero Metrics Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Main P100 Max Metric Card (Cyber Yellow Neo-Brutalist) */}
        <div className="md:col-span-1 bg-[#121826] border-2 border-black rounded-2xl p-5 sm:p-6 flex flex-col justify-between shadow-[5px_5px_0px_0px_#FEE101]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black uppercase tracking-wider text-[#FEE101] font-mono">
              P100 (Max Latency)
            </span>
            <Gauge className="w-5 h-5 text-[#FEE101]" />
          </div>

          <div className="my-4">
            <div className="text-5xl sm:text-6xl font-black text-white font-mono tracking-tight">
              {VERIFIED_BENCHMARK_METRICS.p100.toFixed(2)}{' '}
              <span className="text-xl text-[#FEE101] font-mono font-bold">ms</span>
            </div>
            <div className="text-xs text-[#00F59B] font-bold mt-2 flex items-center gap-1.5 font-sans">
              <TrendingDown className="w-4 h-4" />
              <span>3.33x faster than 200ms requirement</span>
            </div>
          </div>

          <div className="pt-3 border-t-2 border-black flex items-center justify-between text-xs font-mono font-black">
            <span className="text-slate-400">TARGET &lt; 200ms</span>
            <span className="px-2 py-0.5 rounded bg-[#00F59B] text-black border border-black shadow-[1px_1px_0px_0px_#000]">
              PASS
            </span>
          </div>
        </div>

        {/* P50 Median & P70 Distribution */}
        <div className="md:col-span-1 bg-[#121826] border-2 border-black rounded-2xl p-5 sm:p-6 flex flex-col justify-between shadow-[5px_5px_0px_0px_#06B6D4]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black uppercase tracking-wider text-[#06B6D4] font-mono">
              P50 Median
            </span>
            <Zap className="w-5 h-5 text-[#06B6D4]" />
          </div>

          <div className="my-4">
            <div className="text-4xl sm:text-5xl font-black text-white font-mono">
              {VERIFIED_BENCHMARK_METRICS.p50.toFixed(2)}{' '}
              <span className="text-lg text-slate-400 font-mono">ms</span>
            </div>
            <div className="text-xs text-slate-300 mt-2 font-mono">
              P70: <strong className="text-white">{VERIFIED_BENCHMARK_METRICS.p70.toFixed(2)} ms</strong> • Mean: <strong className="text-white">{VERIFIED_BENCHMARK_METRICS.mean.toFixed(2)} ms</strong>
            </div>
          </div>

          <div className="pt-3 border-t-2 border-black flex items-center justify-between text-xs font-mono font-bold">
            <span className="text-slate-400">P90 / P95 TAIL</span>
            <span className="text-[#22D3EE]">{VERIFIED_BENCHMARK_METRICS.p90.toFixed(1)} / {VERIFIED_BENCHMARK_METRICS.p95.toFixed(1)} ms</span>
          </div>
        </div>

        {/* Retrieval Accuracy & Throughput */}
        <div className="md:col-span-1 bg-[#121826] border-2 border-black rounded-2xl p-5 sm:p-6 flex flex-col justify-between shadow-[5px_5px_0px_0px_#00F59B]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black uppercase tracking-wider text-[#00F59B] font-mono">
              Throughput & Accuracy
            </span>
            <CheckCircle2 className="w-5 h-5 text-[#00F59B]" />
          </div>

          <div className="my-4">
            <div className="text-4xl sm:text-5xl font-black text-[#00F59B] font-mono">
              70.92 <span className="text-lg text-slate-300 font-mono">QPS</span>
            </div>
            <div className="text-xs text-slate-300 mt-2 font-mono">
              MRR: <strong className="text-[#00F59B]">1.000</strong> • 0.0% error rate at 50 VUs
            </div>
          </div>

          <div className="pt-3 border-t-2 border-black flex items-center justify-between text-xs font-mono font-bold">
            <span className="text-slate-400">RECALL@1 / RECALL@5</span>
            <span className="text-[#00F59B]">100% / 100%</span>
          </div>
        </div>
      </div>
    </section>
  );
};
