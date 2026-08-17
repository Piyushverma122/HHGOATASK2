export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface ApiStatus {
  online: boolean;
  data?: HealthResponse;
  latencyMs?: number;
  lastChecked?: Date;
  error?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/\/+$/, '')
  : '';

export async function getHealth(): Promise<ApiStatus> {
  const startTime = performance.now();
  try {
    const url = `${API_BASE_URL}/api/v1/health`;
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    const latencyMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      return {
        online: false,
        latencyMs,
        lastChecked: new Date(),
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data: HealthResponse = await response.json();
    return {
      online: data.status === 'ok',
      data,
      latencyMs,
      lastChecked: new Date(),
    };
  } catch (err: unknown) {
    const latencyMs = Math.round(performance.now() - startTime);
    const message = err instanceof Error ? err.message : 'Network error or backend unreachable';
    return {
      online: false,
      latencyMs,
      lastChecked: new Date(),
      error: message,
    };
  }
}
