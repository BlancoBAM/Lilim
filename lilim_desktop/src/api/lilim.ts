/**
 * Lilim API Client — Open Interpreter Backend
 *
 * Connects to OI's FastAPI server using Server-Sent Events (SSE) for streaming.
 * The OI server can be started with:
 *   interpreter --serve (or python -m interpreter.server)
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * OI stream chunk format (LMC Messages)
 */
export interface OIChunk {
  role: 'assistant' | 'computer' | 'user';
  type: 'message' | 'code' | 'console' | 'confirmation' | 'review';
  format?: string;       // "active_line", "output", "python", "javascript", "shell", etc.
  content: string;
  start?: boolean;       // true when a new message type begins
  end?: boolean;         // true when a message type ends
}

/**
 * Assembled message from the stream
 */
export interface LilimMessage {
  id: string;
  role: 'user' | 'assistant' | 'computer';
  type: 'message' | 'code' | 'console' | 'confirmation';
  format?: string;
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
 * Send a chat message via POST and stream the response
 */
export async function* streamChat(
  message: string,
): AsyncGenerator<OIChunk> {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, stream: true }),
    });

    if (!response.ok) {
      throw new LilimAPIError(
        `API request failed: ${response.statusText}`,
        response.status
      );
    }

    if (!response.body) {
      throw new LilimAPIError('No response body — streaming not supported');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Yield a start chunk for the assistant message
    yield {
      role: 'assistant',
      type: 'message',
      content: '',
      start: true
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed.startsWith('data: ')) {
          const jsonStr = trimmed.slice(6);
          if (jsonStr === '[DONE]') continue;
          
          try {
            const data = JSON.parse(jsonStr);
            if (data.type === 'token') {
              yield {
                role: 'assistant',
                type: 'message',
                content: data.text
              };
            } else if (data.type === 'done') {
              yield {
                role: 'assistant',
                type: 'message',
                content: '',
                end: true
              };
            } else if (data.type === 'error') {
               yield {
                 role: 'assistant',
                 type: 'message',
                 content: `\n*Error: ${data.text}*`
               };
            }
          } catch {
            // Not valid JSON, skip
          }
        }
      }
    }

    // Ensure we close the message if it wasn't closed
    yield {
      role: 'assistant',
      type: 'message',
      content: '',
      end: true
    };
  } catch (error) {
    if (error instanceof LilimAPIError) throw error;
    throw new LilimAPIError(
      `Unexpected error: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Send a non-streaming chat message (fallback)
 */
export async function sendMessage(message: string): Promise<OIChunk[]> {
  const chunks: OIChunk[] = [];
  for await (const chunk of streamChat(message)) {
    chunks.push(chunk);
  }
  return chunks;
}

/**
 * Check if the OI backend is reachable
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/heartbeat`, {
      method: 'GET',
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get conversation history from the server
 */
export async function getHistory(): Promise<OIChunk[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/history`, {
      method: 'GET',
    });
    if (!response.ok) return [];
    return response.json();
  } catch {
    return [];
  }
}

/**
 * Get or create a session ID
 */
export function getSessionId(): string {
  let sessionId = localStorage.getItem('lilim_session_id');

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('lilim_session_id', sessionId);
  }

  return sessionId;
}

/**
 * Clear current session
 */
export function clearSession(): void {
  localStorage.removeItem('lilim_session_id');
}
