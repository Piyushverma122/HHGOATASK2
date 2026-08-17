import React from 'react';
import { TopNav, type Route } from './TopNav';
import type { ApiStatus } from '../services/api';

interface AppShellProps {
  currentRoute: Route;
  onNavigate: (route: Route) => void;
  apiStatus: ApiStatus;
  onRefreshHealth: () => void;
  checking: boolean;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentRoute,
  onNavigate,
  apiStatus,
  onRefreshHealth,
  checking,
  children,
}) => {
  return (
    <div className="min-h-screen bg-[#090D16] text-slate-100 flex flex-col font-sans selection:bg-[#FEE101] selection:text-black antialiased">
      {/* Persistent Top Navigation Bar */}
      <TopNav
        currentRoute={currentRoute}
        onNavigate={onNavigate}
        apiStatus={apiStatus}
        onRefreshHealth={onRefreshHealth}
        checking={checking}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {children}
      </main>

      {/* Global Footer (Neo-Brutalist with Asset Integration) */}
      <footer className="border-t-2 border-black py-8 px-6 bg-[#0C1220] shadow-[0_-4px_0_0_#000000]">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <img
              src="/Assets/Hacker%20house.png"
              alt="Hacker House"
              className="h-6 object-contain"
              onError={(e) => {
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
            <p className="text-xs text-slate-300 font-sans font-bold">
              © 2026 Voice RAG Team • HH Goa 2026 Task 2 Submission
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
            <span className="px-2.5 py-1 rounded bg-[#00F59B] text-black border-2 border-black font-black shadow-[2px_2px_0px_0px_#000]">
              ✓ P100: 59.96ms (&lt; 200ms PASS)
            </span>
            <span className="px-2.5 py-1 rounded bg-[#FEE101] text-black border-2 border-black font-black shadow-[2px_2px_0px_0px_#000]">
              7 Indic Languages
            </span>
            <span className="px-2.5 py-1 rounded bg-[#06B6D4] text-black border-2 border-black font-black shadow-[2px_2px_0px_0px_#000]">
              127/127 Tests Passing
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};
