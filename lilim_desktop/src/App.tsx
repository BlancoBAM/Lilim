import { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  isStreaming?: boolean;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I am Lilim, your built-in AI assistant for Lilith Linux. How can I help you today?',
    }
  ]);
  const [input, setInput] = useState('');
  const [isWaiting, setIsWaiting] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isWaiting) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsWaiting(true);

    const assistantMsgId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: 'assistant', content: '', isStreaming: true }
    ]);

    try {
      // Proxy through Rust gateway running on port 8080
      const response = await fetch('http://127.0.0.1:8080/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content, stream: true }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'token') {
                assistantResponse += data.text;
                setMessages((prev) => 
                  prev.map(msg => msg.id === assistantMsgId ? { ...msg, content: assistantResponse } : msg)
                );
              } else if (data.type === 'done' || data.type === 'error') {
                if (data.type === 'error') {
                  assistantResponse += `\n*${data.text}*`;
                }
                setMessages((prev) => 
                  prev.map(msg => msg.id === assistantMsgId ? { ...msg, content: assistantResponse, isStreaming: false } : msg)
                );
                setIsWaiting(false);
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => 
        prev.map(msg => msg.id === assistantMsgId ? { ...msg, content: "*Connection to Lilim Brain failed.*", isStreaming: false } : msg)
      );
      setIsWaiting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Simple Markdown renderer (bold, code blocks, newlines)
  const renderMessageContent = (content: string) => {
    // Check if there is a bash tool block asking for confirmation
    if (content.includes('```bash')) {
      const parts = content.split('```bash');
      const before = parts[0];
      const afterPart = parts[1];
      const commandEndIdx = afterPart.indexOf('```');
      
      if (commandEndIdx !== -1) {
        const command = afterPart.substring(0, commandEndIdx).trim();
        const after = afterPart.substring(commandEndIdx + 3);
        
        return (
          <>
            {renderText(before)}
            <ToolConfirmation command={command} />
            {renderText(after)}
          </>
        );
      }
    }
    return renderText(content);
  };

  const renderText = (text: string) => {
    return text.split('\n').map((line, i) => (
      <span key={i}>
        {line}
        <br />
      </span>
    ));
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#paint0_linear)"/>
            <path d="M2 17L12 22L22 17" stroke="url(#paint1_linear)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="url(#paint2_linear)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <defs>
              <linearGradient id="paint0_linear" x1="12" y1="2" x2="12" y2="12" gradientUnits="userSpaceOnUse">
                <stop stopColor="#6366F1"/>
                <stop offset="1" stopColor="#A855F7"/>
              </linearGradient>
              <linearGradient id="paint1_linear" x1="12" y1="17" x2="12" y2="22" gradientUnits="userSpaceOnUse">
                <stop stopColor="#6366F1"/>
                <stop offset="1" stopColor="#A855F7"/>
              </linearGradient>
              <linearGradient id="paint2_linear" x1="12" y1="12" x2="12" y2="17" gradientUnits="userSpaceOnUse">
                <stop stopColor="#6366F1"/>
                <stop offset="1" stopColor="#A855F7"/>
              </linearGradient>
            </defs>
          </svg>
          Lilim
        </h1>
        <div className="header-status">
          <div className="status-dot"></div>
          Online
        </div>
      </header>

      {/* Chat Messages */}
      <main className="chat-container">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.role}`}>
            <div className={`message ${msg.role}`}>
              {msg.role === 'assistant' && msg.isStreaming && msg.content === '' ? (
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              ) : (
                <div className="message-content">
                  {renderMessageContent(msg.content)}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </main>

      {/* Input Area */}
      <footer className="input-container">
        <div className="input-box">
          <textarea
            className="input-field"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Lilim..."
            rows={1}
            disabled={isWaiting}
          />
          <button className="send-button" onClick={handleSend} disabled={isWaiting || !input.trim()}>
            <svg viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </footer>
    </div>
  );
}

function ToolConfirmation({ command }: { command: string }) {
  const [status, setStatus] = useState<'pending' | 'running' | 'success' | 'error'>('pending');
  const [result, setResult] = useState<{stdout: string, stderr: string} | null>(null);

  const executeCommand = async () => {
    setStatus('running');
    try {
      const response = await fetch('http://127.0.0.1:8080/tools/shell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, confirmed: true }),
      });
      
      const data = await response.json();
      setResult({ stdout: data.stdout, stderr: data.stderr || data.error });
      setStatus(data.returncode === 0 ? 'success' : 'error');
    } catch (e: any) {
      setResult({ stdout: '', stderr: e.message });
      setStatus('error');
    }
  };

  return (
    <div className="tool-card">
      <div className="tool-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="4 17 10 11 4 5"></polyline>
          <line x1="12" y1="19" x2="20" y2="19"></line>
        </svg>
        System Command Requested
      </div>
      <div className="tool-command">{command}</div>
      
      {status === 'pending' && (
        <div className="tool-actions">
          <button className="btn btn-primary" onClick={executeCommand}>Run it</button>
        </div>
      )}
      
      {status === 'running' && (
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Executing...</div>
      )}
      
      {result && (
        <div className={`tool-execution-result ${status}`}>
          <div className="tool-execution-header">
            <span>Result</span>
            <span>{status === 'success' ? '✓ Success' : '✗ Failed'}</span>
          </div>
          <div className="tool-execution-output">
            {result.stdout && <div>{result.stdout}</div>}
            {result.stderr && <div style={{ color: 'var(--danger)' }}>{result.stderr}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
