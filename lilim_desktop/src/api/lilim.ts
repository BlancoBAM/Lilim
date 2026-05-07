/**
 * Lilim API Client — Native Rust Gateway Backend
 *
 * Connects to lilim-runtime (Rust proxy) on port 8080 via SSE streaming.
 * The gateway proxies to the Python FastAPI brain on port 8081.
 * Local inference (Phi-2) is handled directly in the Rust gateway.
 */

const API_BASE_URL = 'http://127.0.0.1:8080';

export interface LilimMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  provider?: string;
}

export class LilimAPIError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'LilimAPIError';
  }
}

export interface OIChunk {
  role: 'assistant';
  type: 'message';
  content: string;
  start?: boolean;
  end?: boolean;
  provider?: string;
}

export interface ProviderStatus {
  name: string;
  configured: boolean;
  daily_limit: number;
  tokens_per_min: number;
  failures: number;
  free_models: string[];
}

export interface ModelStatus {
  local_engine: {
    available: boolean;
    model: string;
    device: string;
    model_status: {
      available: boolean;
      location: string;
      source: string;
      size_mb: number;
    };
  };
}

/**
 * Stream a chat response from the Rust gateway (SSE).
 * Yields OIChunk objects compatible with the ChatInterface.
 */
export async function* streamChat(message: string, signal?: AbortSignal): AsyncGenerator<OIChunk> {
  const sessionId = getSessionId();

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, stream: true }),
      signal,
    });
  } catch (err) {
    if ((err as any).name === 'AbortError') throw err;
    throw new LilimAPIError(
      `Cannot connect to Lilim backend at ${API_BASE_URL}. Is the lilith-ai service running? ` +
      `Run: systemctl start lilith-ai`
    );
  }

  if (!response.ok) {
    throw new LilimAPIError(`API error ${response.status}: ${response.statusText}`, response.status);
  }

  if (!response.body) {
    throw new LilimAPIError('No response body — streaming not supported');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  yield { role: 'assistant', type: 'message', content: '', start: true };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data: ')) continue;

        const jsonStr = trimmed.slice(6);
        if (jsonStr === '[DONE]') continue;

        try {
          const data = JSON.parse(jsonStr);
          if (data.type === 'token' && data.text) {
            yield { role: 'assistant', type: 'message', content: data.text };
          } else if (data.type === 'status') {
            // Agentic status — show as italic
            yield { role: 'assistant', type: 'message', content: `\n*${data.text}*\n` };
          } else if (data.type === 'done') {
            yield { role: 'assistant', type: 'message', content: '', end: true, provider: data.provider };
            return;
          } else if (data.type === 'error') {
            yield { role: 'assistant', type: 'message', content: `\n*${data.text}*` };
          }
        } catch {
          // Non-JSON SSE line, skip
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield { role: 'assistant', type: 'message', content: '', end: true };
}

/**
 * Execute a confirmed shell command via the Rust security gateway.
 */
export async function runShellCommand(command: string): Promise<{
  stdout: string;
  stderr: string;
  returncode: number;
}> {
  const response = await fetch(`${API_BASE_URL}/tools/shell`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command, confirmed: true }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new LilimAPIError(`Shell command rejected: ${detail}`, response.status);
  }
  return response.json();
}

/**
 * Get model and inference engine status.
 */
export async function getModelStatus(): Promise<ModelStatus | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/model/status`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Get all provider statuses (which are configured, rate limit info, etc.)
 */
export async function getProvidersStatus(): Promise<{ providers: ProviderStatus[]; configured_count: number } | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/providers/status`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Register an API key with optional provider hint.
 * The backend auto-detects the provider from the key format.
 */
export async function registerApiKey(
  apiKey: string,
  provider?: string,
  model?: string
): Promise<{ status: string; provider: string } | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/providers/register-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey, provider, model }),
    });
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Save model config to backend (hot-reload).
 */
export async function saveModelConfig(config: Record<string, unknown>): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/settings/model-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
  } catch {
    // best-effort
  }
}

/**
 * Check if the backend is reachable.
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export function getSessionId(): string {
  let sessionId = localStorage.getItem('lilim_session_id');
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    localStorage.setItem('lilim_session_id', sessionId);
  }
  return sessionId;
}

export function clearSession(): void {
  localStorage.removeItem('lilim_session_id');
}
