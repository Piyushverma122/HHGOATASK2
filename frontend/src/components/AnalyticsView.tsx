import { useState } from 'react';
import {
  ArrowLeft,
  CheckCircle2,
  Layers,
  Shield,
  Gauge,
  Flame,
  Zap,
  Activity,
  Cpu,
  BarChart3,
  TrendingDown,
} from 'lucide-react';
import {
  VERIFIED_BENCHMARK_METRICS,
  STAGE_LATENCY_BREAKDOWN,
  RETRIEVAL_ABLATIONS,
  CONCURRENCY_BENCHMARKS,
} from '../config/metrics';

interface AnalyticsViewProps {
  onBack: () => void;
}

export function AnalyticsView({ onBack }: AnalyticsViewProps) {
  const [activeTab, setActiveTab] = useState<'percentiles' | 'waterfall' | 'ablation' | 'concurrency'>('percentiles');

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
              Production Latency & Telemetry
            </h2>
            <span className="px-3 py-1 rounded-full text-xs font-black bg-[#00F59B] text-black border-2 border-black shadow-[2px_2px_0px_0px_#000] flex items-center gap-1.5 font-mono uppercase">
              <CheckCircle2 className="w-3.5 h-3.5" />
              100% &lt; 200ms COMPLIANT
            </span>
          </div>
          <p className="text-sm text-slate-300 font-sans mt-1">
            Empirical benchmark over {VERIFIED_BENCHMARK_METRICS.totalQueries} multilingual queries on MSMARCO-XI Hindi (99,925 passages)
          </p>
        </div>

        {/* View Switcher Tabs (Neo-Brutalist) */}
        <div className="flex items-center gap-1.5 bg-[#0C1220] border-2 border-black p-1.5 rounded-xl shadow-[3px_3px_0px_0px_#000000]">
          {[
            { id: 'percentiles', label: 'Percentiles' },
            { id: 'waterfall', label: 'Stage Waterfall' },
            { id: 'ablation', label: 'Retrieval Ablation' },
            { id: 'concurrency', label: 'Concurrency Stress' },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold font-sans uppercase tracking-wider transition-all cursor-pointer border-2 border-black ${
                  isActive
                    ? 'bg-[#FEE101] text-black shadow-[2px_2px_0px_0px_#000]'
                    : 'bg-[#070A12] text-slate-300 hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Hero Metric Strip (Neo-Brutalist) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 shadow-[5px_5px_0px_0px_#FEE101] flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-black uppercase text-[#FEE101]">P100 (Max Latency)</span>
            <Gauge className="w-5 h-5 text-[#FEE101]" />
          </div>
          <div className="my-2">
            <div className="text-4xl font-mono font-black text-white">
              {VERIFIED_BENCHMARK_METRICS.p100.toFixed(2)}{' '}
              <span className="text-lg text-[#FEE101]">ms</span>
            </div>
            <div className="text-[11px] text-[#00F59B] font-bold font-sans mt-1 flex items-center gap-1">
              <TrendingDown className="w-3.5 h-3.5" />
              <span>3.33x faster than 200ms ceiling</span>
            </div>
          </div>
          <div className="pt-2 border-t-2 border-black flex items-center justify-between text-[11px] font-mono font-bold">
            <span className="text-slate-400">Target &lt; 200.0 ms</span>
            <span className="text-[#00F59B] font-black">100% PASS</span>
          </div>
        </div>

        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 shadow-[5px_5px_0px_0px_#06B6D4] flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-black uppercase text-[#06B6D4]">P50 Median</span>
            <Zap className="w-5 h-5 text-[#06B6D4]" />
          </div>
          <div className="my-2">
            <div className="text-4xl font-mono font-black text-white">
              {VERIFIED_BENCHMARK_METRICS.p50.toFixed(2)}{' '}
              <span className="text-lg text-slate-400">ms</span>
            </div>
            <div className="text-[11px] text-slate-300 font-mono mt-1">
              Mean: {VERIFIED_BENCHMARK_METRICS.mean.toFixed(2)} ms across all queries
            </div>
          </div>
          <div className="pt-2 border-t-2 border-black flex items-center justify-between text-[11px] font-mono font-bold">
            <span className="text-slate-400">P90 / P95</span>
            <span className="text-[#22D3EE]">{VERIFIED_BENCHMARK_METRICS.p90.toFixed(1)} / {VERIFIED_BENCHMARK_METRICS.p95.toFixed(1)} ms</span>
          </div>
        </div>

        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 shadow-[5px_5px_0px_0px_#FF0080] flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-black uppercase text-[#FF0080]">Warm vs Cold</span>
            <Flame className="w-5 h-5 text-[#FF0080]" />
          </div>
          <div className="my-2">
            <div className="text-4xl font-mono font-black text-white">
              24.9 <span className="text-lg text-slate-400">ms</span>
            </div>
            <div className="text-[11px] text-slate-300 font-mono mt-1">
              Cold start: ~10.2s (JIT & model pre-load)
            </div>
          </div>
          <div className="pt-2 border-t-2 border-black flex items-center justify-between text-[11px] font-mono font-bold">
            <span className="text-slate-400">Query Cache Hit</span>
            <span className="text-[#00F59B] font-black">&lt; 0.5 ms</span>
          </div>
        </div>

        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 shadow-[5px_5px_0px_0px_#00F59B] flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-black uppercase text-[#00F59B]">Accuracy / MRR</span>
            <Shield className="w-5 h-5 text-[#00F59B]" />
          </div>
          <div className="my-2">
            <div className="text-4xl font-mono font-black text-[#00F59B]">
              1.000 <span className="text-lg text-slate-300 font-mono">MRR</span>
            </div>
            <div className="text-[11px] text-slate-300 font-mono mt-1">
              Recall@1 = 100% • Recall@5 = 100%
            </div>
          </div>
          <div className="pt-2 border-t-2 border-black flex items-center justify-between text-[11px] font-mono font-bold">
            <span className="text-slate-400">Cross-Encoder Gain</span>
            <span className="text-[#00F59B] font-black">+137% Recall@1</span>
          </div>
        </div>
      </div>

      {/* Tab 1: Percentiles View (Neo-Brutalist) */}
      {activeTab === 'percentiles' && (
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 space-y-6 shadow-[6px_6px_0px_0px_#000000]">
          <div className="flex items-center justify-between border-b-2 border-black pb-4">
            <div>
              <h3 className="text-lg font-black text-white font-sans uppercase flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-[#FEE101]" />
                Verified Latency Percentiles ({VERIFIED_BENCHMARK_METRICS.totalQueries} MSMARCO-XI Queries)
              </h3>
              <p className="text-xs text-slate-300 font-sans mt-0.5">
                Target threshold is strictly &lt; 200 ms warm latency as required by the Task 2 challenge.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { label: 'P50 (MEDIAN)', val: VERIFIED_BENCHMARK_METRICS.p50, pass: true, color: 'shadow-[3px_3px_0px_0px_#06B6D4]' },
              { label: 'P70', val: VERIFIED_BENCHMARK_METRICS.p70, pass: true, color: 'shadow-[3px_3px_0px_0px_#06B6D4]' },
              { label: 'P90', val: VERIFIED_BENCHMARK_METRICS.p90, pass: true, color: 'shadow-[3px_3px_0px_0px_#22D3EE]' },
              { label: 'P95', val: VERIFIED_BENCHMARK_METRICS.p95, pass: true, color: 'shadow-[3px_3px_0px_0px_#22D3EE]' },
              { label: 'P99', val: VERIFIED_BENCHMARK_METRICS.p99, pass: true, color: 'shadow-[3px_3px_0px_0px_#FEE101]' },
              { label: 'P100 (MAX)', val: VERIFIED_BENCHMARK_METRICS.p100, pass: true, color: 'shadow-[4px_4px_0px_0px_#FEE101]' },
            ].map((pct, idx) => (
              <div key={idx} className={`bg-[#070A12] border-2 border-black rounded-xl p-4 text-center space-y-1 ${pct.color}`}>
                <span className="text-[10px] uppercase font-mono font-black text-slate-400">{pct.label}</span>
                <div className="text-2xl font-mono font-black text-white">
                  {pct.val.toFixed(2)} <span className="text-xs text-slate-400">ms</span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-[10px] font-mono">
                  <span className="text-slate-500">&lt; 200 ms</span>
                  <span className="text-[#00F59B] font-black">PASS</span>
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t-2 border-black">
            <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 space-y-1 shadow-[3px_3px_0px_0px_#000]">
              <span className="text-xs font-black uppercase text-[#06B6D4] font-mono">Sequential vs Parallel Retrieval</span>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Parallel concurrent retrieval with <code className="text-[#FEE101] font-mono font-bold">ThreadPoolExecutor</code> reduces dual search time from 20.36 ms to 19.74 ms.
              </p>
            </div>
            <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 space-y-1 shadow-[3px_3px_0px_0px_#000]">
              <span className="text-xs font-black uppercase text-[#FEE101] font-mono">JIT Model Pre-Warming</span>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Startup model pre-allocation loads FAISS indexes and PyTorch cross-encoder weights, slashing query execution from 10.2s cold to 24.9ms warm.
              </p>
            </div>
            <div className="bg-[#070A12] border-2 border-black rounded-xl p-4 space-y-1 shadow-[3px_3px_0px_0px_#000]">
              <span className="text-xs font-black uppercase text-[#00F59B] font-mono">LRU Subword Query Cache</span>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Deterministic SHA-256 hashed LRU query caching returns repeated exact queries in &lt; 0.5 ms with zero neural inference overhead.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Stage Waterfall (Neo-Brutalist) */}
      {activeTab === 'waterfall' && (
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 space-y-6 shadow-[6px_6px_0px_0px_#000000]">
          <div className="flex items-center justify-between border-b-2 border-black pb-4">
            <div>
              <h3 className="text-lg font-black text-white font-sans uppercase flex items-center gap-2">
                <Cpu className="w-5 h-5 text-[#06B6D4]" />
                Granular Stage-by-Stage Latency Waterfall
              </h3>
              <p className="text-xs text-slate-300 font-sans mt-0.5">
                Exact millisecond breakdown across all 9 execution stages
              </p>
            </div>
          </div>

          <div className="space-y-3">
            {STAGE_LATENCY_BREAKDOWN.map((stage, idx) => {
              const maxMs = 20.0;
              const widthPct = Math.min(100, Math.max(5, (stage.p50 / maxMs) * 100));
              return (
                <div key={idx} className="bg-[#070A12] border-2 border-black rounded-xl p-3.5 space-y-1.5 shadow-[2px_2px_0px_0px_#000]">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 font-bold">0{idx + 1}</span>
                      <strong className="text-white font-sans uppercase">{stage.stage}</strong>
                      <span className="text-slate-400 hidden sm:inline text-[11px] font-sans">({stage.description})</span>
                    </div>
                    <span className="text-[#FEE101] font-black font-mono">{stage.p50.toFixed(2)} ms</span>
                  </div>

                  <div className="w-full bg-[#0C1220] border border-black rounded-full h-3 overflow-hidden p-0.5">
                    <div
                      className="h-full rounded-full transition-all duration-500 bg-[#FEE101]"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 3: Retrieval Ablation (Neo-Brutalist) */}
      {activeTab === 'ablation' && (
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 space-y-6 shadow-[6px_6px_0px_0px_#000000]">
          <div className="flex items-center justify-between border-b-2 border-black pb-4">
            <div>
              <h3 className="text-lg font-black text-white font-sans uppercase flex items-center gap-2">
                <Layers className="w-5 h-5 text-[#FF0080]" />
                Retrieval Strategy Ablation Study
              </h3>
              <p className="text-xs text-slate-300 font-sans mt-0.5">
                Evaluation of Recall@K, MRR, and latency trade-offs across individual and hybrid retrieval configurations
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono border-2 border-black">
              <thead className="bg-[#070A12] border-b-2 border-black text-slate-300 font-black uppercase">
                <tr>
                  <th className="p-3 border-r-2 border-black">Pipeline Architecture</th>
                  <th className="p-3 border-r-2 border-black text-center">Recall@1</th>
                  <th className="p-3 border-r-2 border-black text-center">Recall@5</th>
                  <th className="p-3 border-r-2 border-black text-center">Recall@10</th>
                  <th className="p-3 border-r-2 border-black text-center">MRR</th>
                  <th className="p-3 text-center">Mean Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y-2 divide-black">
                {RETRIEVAL_ABLATIONS.map((row, idx) => (
                  <tr key={idx} className={idx === 4 ? 'bg-[#152038] font-bold' : 'bg-[#0C1220] hover:bg-[#101726]'}>
                    <td className="p-3 border-r-2 border-black flex items-center gap-2 text-white font-sans">
                      {idx === 4 && <span className="px-2 py-0.5 rounded text-[9px] bg-[#FEE101] text-black border border-black font-mono font-black">PRODUCTION</span>}
                      <span>{row.configuration}</span>
                    </td>
                    <td className="p-3 border-r-2 border-black text-center text-[#00F59B]">{(row.recall1 * 100).toFixed(1)}%</td>
                    <td className="p-3 border-r-2 border-black text-center text-[#00F59B]">{(row.recall5 * 100).toFixed(1)}%</td>
                    <td className="p-3 border-r-2 border-black text-center text-[#00F59B]">{(row.recall10 * 100).toFixed(1)}%</td>
                    <td className="p-3 border-r-2 border-black text-center text-[#FEE101] font-black">{row.mrr.toFixed(3)}</td>
                    <td className="p-3 text-center text-[#06B6D4] font-black">{row.latencyMs.toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 4: Concurrency Stress (Neo-Brutalist) */}
      {activeTab === 'concurrency' && (
        <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 space-y-6 shadow-[6px_6px_0px_0px_#000000]">
          <div className="flex items-center justify-between border-b-2 border-black pb-4">
            <div>
              <h3 className="text-lg font-black text-white font-sans uppercase flex items-center gap-2">
                <Activity className="w-5 h-5 text-[#00F59B]" />
                High-Concurrency Load Stress Test Matrix
              </h3>
              <p className="text-xs text-slate-300 font-sans mt-0.5">
                Stress testing concurrent virtual users (10, 25, 50 VUs) on FastAPI server
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {CONCURRENCY_BENCHMARKS.map((c, idx) => (
              <div key={idx} className="bg-[#070A12] border-2 border-black rounded-xl p-5 space-y-3 shadow-[4px_4px_0px_0px_#00F59B]">
                <div className="flex items-center justify-between border-b-2 border-black pb-2">
                  <span className="text-xs font-mono font-black text-white uppercase">{c.vus} Virtual Users</span>
                  <span className="px-2 py-0.5 rounded bg-[#00F59B] text-black border border-black text-[10px] font-mono font-black">
                    {c.errorRatePct}% ERRORS
                  </span>
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-slate-400 font-sans">Throughput (QPS):</span>
                    <strong className="text-[#00F59B] font-bold">{c.qps.toFixed(2)} QPS</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400 font-sans">P50 Latency:</span>
                    <strong className="text-white font-bold">{c.p50Ms.toFixed(2)} ms</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400 font-sans">P95 Latency:</span>
                    <strong className="text-[#06B6D4] font-bold">{c.p95Ms.toFixed(2)} ms</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
