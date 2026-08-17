import { useEffect, useState, useCallback } from 'react';
import { getHealth, type ApiStatus } from './services/api';
import { AppShell } from './components/AppShell';
import type { Route } from './components/TopNav';
import { DashboardView } from './components/DashboardView';
import { VoiceStudioView } from './components/VoiceStudioView';
import { GuardrailDemoView } from './components/GuardrailDemoView';
import { AnalyticsView } from './components/AnalyticsView';

export function App() {
  const [currentRoute, setCurrentRoute] = useState<Route>(() => {
    const path = window.location.pathname;
    if (['/voice', '/guardrails', '/analytics'].includes(path)) {
      return path as Route;
    }
    return '/';
  });

  const [apiStatus, setApiStatus] = useState<ApiStatus>({ online: false });
  const [checking, setChecking] = useState<boolean>(false);

  const checkBackendHealth = useCallback(async () => {
    setChecking(true);
    const status = await getHealth();
    setApiStatus(status);
    setChecking(false);
  }, []);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 10000);
    return () => clearInterval(interval);
  }, [checkBackendHealth]);

  const navigateTo = (route: Route) => {
    setCurrentRoute(route);
    window.history.pushState({}, '', route);
  };

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname;
      if (['/voice', '/guardrails', '/analytics'].includes(path)) {
        setCurrentRoute(path as Route);
      } else {
        setCurrentRoute('/');
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  return (
    <AppShell
      currentRoute={currentRoute}
      onNavigate={navigateTo}
      apiStatus={apiStatus}
      onRefreshHealth={checkBackendHealth}
      checking={checking}
    >
      {currentRoute === '/' && (
        <DashboardView
          apiStatus={apiStatus}
          onNavigate={navigateTo}
          onRefresh={checkBackendHealth}
          checking={checking}
        />
      )}
      {currentRoute === '/voice' && <VoiceStudioView onBack={() => navigateTo('/')} />}
      {currentRoute === '/guardrails' && <GuardrailDemoView onBack={() => navigateTo('/')} />}
      {currentRoute === '/analytics' && <AnalyticsView onBack={() => navigateTo('/')} />}
    </AppShell>
  );
}

export default App;
