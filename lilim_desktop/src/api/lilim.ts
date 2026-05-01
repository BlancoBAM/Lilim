/**
 * Lilim API Client — Native Rust Gateway Backend
 *
 * Connects to lilim-runtime (Rust proxy) on port 8080 via SSE streaming.
 * The gateway proxies to the Python FastAPI brain on port 8081.
 */

// The Rust gateway always runs on 8080 on localhost
const API_BASE_URL = 'http://127.0.0.1:8080';

export interface LilimMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export class LilimAPIError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'LilimAPIError';
  }
}

/**
 * Chunk yielded from the stream to the UI
 */
export interface OIChunk {
  role: 'assistant';
  type: 'message';
  content: string;
  start?: boolean;
  end?: boolean;
}

/**
 * Stream a chat response from the Rust gateway (SSE).
 * Yields OIChunk objects compatible with the legacy ChatInterface.
 */
export async function* streamChat(message: string): AsyncGenerator<OIChunk> {
  // Retrieve or create a session ID so memory is per-session
  const sessionId = getSessionId();

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, stream: true }),
    });
  } catch (err) {
    throw new LilimAPIError(
      `Cannot connect to Lilim backend at ${API_BASE_URL}. Is the lilith-ai service running?`
    );
  }

  if (!response.ok) {
    throw new LilimAPIError(
      `API error ${response.status}: ${response.statusText}`,
      response.status
    );
  }

  if (!response.body) {
    throw new LilimAPIError('No response body — streaming not supported by this client');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // Signal start of assistant message to the UI
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
          } else if (data.type === 'done') {
            yield { role: 'assistant', type: 'message', content: '', end: true };
            return;
          } else if (data.type === 'error') {
            yield { role: 'assistant', type: 'message', content: `\n*${data.text}*` };
          }
          // 'meta' chunks (model info) are silently ignored
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

/**
 * Get or create a persistent session ID stored in localStorage.
 */
export function getSessionId(): string {
  let sessionId = localStorage.getItem('lilim_session_id');
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    localStorage.setItem('lilim_session_id', sessionId);
  }
  return sessionId;
}

/** Clear the current session (new conversation). */
export function clearSession(): void {
  localStorage.removeItem('lilim_session_id');
}
