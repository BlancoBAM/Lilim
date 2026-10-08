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
  pending_command?: string;
  pending_short?: string;
  pending_sudo?: boolean;
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
 * Detect the real logged-in desktop user.
 * In Tauri, the process runs as the actual user, so env vars are authoritative.
 * We cache after the first read so subsequent calls are free.
 */
let _cachedUserCtx: { username: string; home_dir: string } | null = null;
function getUserContext(): { username: string; home_dir: string } {
  if (_cachedUserCtx) return _cachedUserCtx;
  // In Tauri/Electron the renderer can read env via import.meta.env or process.env.
  // Vite exposes VITE_* vars; for runtime system vars we use a small heuristic:
  // Try window.__TAURI_INTERNALS__ metadata, then fall back to navigator clues.
  const username =
    (window as any).__LILIM_USER__ ||
    document.cookie.match(/lilim_user=([^;]+)/)?.[1] ||
    localStorage.getItem('lilim_detected_user') ||
    '';
  const home_dir =
    (window as any).__LILIM_HOME__ ||
    localStorage.getItem('lilim_detected_home') ||
    '';
  _cachedUserCtx = { username, home_dir };
  return _cachedUserCtx;
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
      body: JSON.stringify({
        message,
        session_id: sessionId,
        stream: true,
        ...getUserContext(),  // sends username + home_dir so backend knows real desktop user
      }),
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
          } else if (data.type === 'tool_call') {
            // Auto-executed command from the ReAct agent loop — show as inline status
            yield { role: 'assistant', type: 'message', content: `\n*⚡ Executing: \`${data.text}\`*\n` };
          } else if (data.type === 'tool_pending') {
            // Command needs user confirmation — bubble this up as a special chunk
            yield {
              role: 'assistant',
              type: 'message',
              content: '',
              // @ts-ignore — extend chunk with pending command data
              pending_command: data.command,
              pending_short: data.short,
              pending_sudo: data.sudo === true,
              end: true,
              provider: 'PENDING',
            };
            return;
          } else if (data.type === 'status') {
            // Legacy status messages
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
export async function runShellCommand(
  command: string,
  sudo?: { sessionId: string; password: string },
): Promise<{
  stdout: string;
  stderr: string;
  returncode: number;
}> {
  const response = await fetch(`${API_BASE_URL}${sudo ? '/tools/shell/sudo' : '/tools/shell'}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sudo
      ? { command, confirmed: true, session_id: sudo.sessionId, password: sudo.password }
      : { command, confirmed: true }),
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
 * Get model config from backend.
 */
export async function getModelConfig(): Promise<Record<string, string>> {
  try {
    const response = await fetch(`${API_BASE_URL}/settings/model-config`);
    if (!response.ok) return {};
    return response.json();
  } catch {
    return {};
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

export interface UserProfile {
  system_username: string;
  system_home: string;
  display_name: string;
  github_username: string;
  preferred_home: string;
}

/**
 * Get the current user profile (system-detected + any overrides).
 */
export async function getUserProfile(): Promise<UserProfile | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/settings/user-profile`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Save user profile overrides (display_name, github_username, preferred_home).
 */
export async function saveUserProfile(
  profile: Partial<Pick<UserProfile, 'display_name' | 'github_username' | 'preferred_home'>>
): Promise<{ status: string; profile: UserProfile } | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/settings/user-profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
    });
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

/**
 * Send a shell command observation back into the LLM conversation.
 * Call this after a confirmed shell command completes so the AI sees the real output.
 */
export async function* sendObservation(
  observationText: string,
  signal?: AbortSignal
): AsyncGenerator<OIChunk> {
  const message = `Observation: ${observationText}`;
  yield* streamChat(message, signal);
}
