import React, { useState } from 'react';
import {
  Mic,
  Activity,
  Shield,
  Layers,
  RefreshCw,
  Menu,
  X,
} from 'lucide-react';
import type { ApiStatus } from '../services/api';

export type Route = '/' | '/voice' | '/retrieval' | '/guardrails' | '/analytics';

interface TopNavProps {
  currentRoute: Route;
  onNavigate: (route: Route) => void;
  apiStatus: ApiStatus;
  onRefreshHealth: () => void;
  checking: boolean;
}

export const TopNav: React.FC<TopNavProps> = ({
  currentRoute,
  onNavigate,
  apiStatus,
  onRefreshHealth,
  checking,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: { route: Route; label: string; icon: React.ReactNode }[] = [
    { route: '/', label: 'Dashboard', icon: null },
    { route: '/voice', label: 'Voice Studio', icon: <Mic className="w-3.5 h-3.5" /> },
    { route: '/retrieval', label: 'Retrieval', icon: <Layers className="w-3.5 h-3.5" /> },
    { route: '/guardrails', label: 'Guardrails', icon: <Shield className="w-3.5 h-3.5" /> },
    { route: '/analytics', label: 'Latency', icon: <Activity className="w-3.5 h-3.5" /> },
  ];

  const handleNav = (route: Route) => {
    onNavigate(route);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-[#0C1220] border-b-2 border-black px-4 sm:px-6 lg:px-8 py-3 shadow-[0_4px_0_0_#000000]">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand & Identity (Neo-Brutalist Badge) */}
        <div
          onClick={() => handleNav('/')}
          className="flex items-center gap-3 cursor-pointer group shrink-0"
        >
          <div className="relative w-11 h-11 rounded-xl bg-[#FEE101] border-2 border-black flex items-center justify-center shadow-[3px_3px_0px_0px_#000000] group-hover:translate-x-0.5 group-hover:translate-y-0.5 group-hover:shadow-[1px_1px_0px_0px_#000000] transition-all">
            <img
              src="/Assets/goa_hindi.svg"
              alt="Goa"
              className="w-8 h-8 object-contain"
              onError={(e) => {
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-black tracking-wider text-white uppercase font-sans">
                VOICE RAG
              </h1>
              <span className="text-[9px] uppercase font-mono font-black px-2 py-0.5 rounded bg-[#FF0080] text-white border-2 border-black shadow-[2px_2px_0px_0px_#000000]">
                HH GOA 2026
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden lg:block font-mono">
              Multilingual Grounded RAG • Sarvam STT • Hybrid Retrieval
            </p>
          </div>
        </div>

        {/* Center Desktop Navigation Tabs (Neo-Brutalist Buttons) */}
        <nav className="hidden md:flex items-center gap-1.5 bg-[#070A12] border-2 border-black p-1.5 rounded-xl shadow-[3px_3px_0px_0px_#000000]">
          {navItems.map((item) => {
            const isActive = currentRoute === item.route;
            return (
              <button
                key={item.route}
                onClick={() => handleNav(item.route)}
                className={`relative px-3.5 py-1.5 rounded-lg text-xs font-bold font-sans uppercase tracking-wider transition-all flex items-center gap-1.5 cursor-pointer select-none ${
                  isActive
                    ? 'bg-[#FEE101] text-black border-2 border-black shadow-[2px_2px_0px_0px_#000000] -translate-x-0.5 -translate-y-0.5'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/80 border-2 border-transparent'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Status & Mobile Toggle */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={onRefreshHealth}
            disabled={checking}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border-2 border-black text-xs font-mono font-bold transition-all shadow-[3px_3px_0px_0px_#000000] cursor-pointer ${
              apiStatus.online ? 'bg-[#00F59B] text-black' : 'bg-[#FF3366] text-white'
            }`}
            title="Click to check backend connectivity"
          >
            <span className="w-2 h-2 rounded-full bg-black shrink-0" />
            <span className="hidden sm:inline text-[11px]">
              {apiStatus.online ? 'CONNECTED (~24.9ms)' : 'DEMO / OFFLINE'}
            </span>
            <RefreshCw className={`w-3 h-3 ${checking ? 'animate-spin' : ''}`} />
          </button>

          {/* Mobile Hamburger Toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-xl bg-[#070A12] border-2 border-black text-white shadow-[2px_2px_0px_0px_#000000]"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#0C1220] border-t-2 border-black mt-3 pt-3 px-2 pb-2 space-y-1.5 animate-fadeIn">
          {navItems.map((item) => {
            const isActive = currentRoute === item.route;
            return (
              <button
                key={item.route}
                onClick={() => handleNav(item.route)}
                className={`w-full px-4 py-2.5 rounded-xl text-xs font-bold font-sans uppercase tracking-wider flex items-center gap-2.5 transition border-2 border-black ${
                  isActive
                    ? 'bg-[#FEE101] text-black shadow-[3px_3px_0px_0px_#000000]'
                    : 'bg-[#070A12] text-slate-200 hover:text-white'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};
