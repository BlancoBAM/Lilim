import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, X, Minus, Flame } from 'lucide-react';
import { FlameBackground } from './FlameBackground';
import { EmberOverlay } from './EmberOverlay';
import bannerImage from '../assets/c80b4d356e3c7b98f2baabf558ea7bacc2421ec9.png';
import centerLogo from '../assets/03a17ee9fd4fe33c3ca16baf528b1598cfae5797.png';
import topLeftLogo from '../assets/51350c1f0fe5a2742ba35cd8899037600d9d9f62.png';
import { streamChat, runShellCommand, type LilimMessage } from '../api/lilim';
import { getCurrentWindow } from '@tauri-apps/api/window';

const appWindow = getCurrentWindow();

export function ChatInterface() {
  const [messages, setMessages] = useState<LilimMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        'Greetings, seeker. I am Lilim, your guide through the flames of knowledge. What wisdom do you seek today?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* ── Window controls ── */
  const handleClose = () => appWindow.close();
  const handleMinimize = () => appWindow.minimize();

  /* ── Send message ── */
  const handleSend = useCallback(async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: LilimMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsStreaming(true);

    const assistantId = (Date.now() + 1).toString();

    try {
      let accumulated = '';

      for await (const chunk of streamChat(userMessage.content)) {
        if (chunk.start) {
          setMessages(prev => [
            ...prev,
            { id: assistantId, role: 'assistant', content: '', timestamp: new Date() },
          ]);
          continue;
        }
        if (chunk.end) continue;
        if (!chunk.content) continue;

        accumulated += chunk.content;
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, content: accumulated } : m))
        );
      }
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content:
            error instanceof Error
              ? `*The flames flicker... ${error.message}*`
              : '*The flames flicker... An unknown error occurred.*',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsStreaming(false);
    }
  }, [input, isStreaming]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ── Shell command confirmation ── */
  const handleRunCommand = async (command: string) => {
    try {
      const result = await runShellCommand(command);
      const output = result.stdout || result.stderr || '(no output)';
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: `\`\`\`\n${output.trim()}\n\`\`\``,
          timestamp: new Date(),
        },
      ]);
    } catch (e) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: `*Command failed: ${e instanceof Error ? e.message : String(e)}*`,
          timestamp: new Date(),
        },
      ]);
    }
  };

  /* ── Render a single message bubble ── */
  const renderContent = (message: LilimMessage) => {
    if (message.role === 'user') {
      return <p className="whitespace-pre-wrap relative z-10">{message.content}</p>;
    }

    // Parse ```bash blocks to show confirmation UI
    const content = message.content;
    const bashMatch = content.match(/```bash\n?([\s\S]*?)```/);
    if (bashMatch) {
      const [fullMatch, command] = bashMatch;
      const before = content.slice(0, content.indexOf(fullMatch));
      const after = content.slice(content.indexOf(fullMatch) + fullMatch.length);

      return (
        <div className="relative z-10 space-y-2">
          {before && <p className="whitespace-pre-wrap">{before.trim()}</p>}
          <div className="bg-black/40 border border-orange-500/40 rounded-lg p-3">
            <p className="text-orange-300 text-xs mb-2 flex items-center gap-1">
              <Flame size={12} /> System command requested:
            </p>
            <pre className="bg-gray-950/80 text-green-300 p-2 rounded text-xs font-mono mb-3 overflow-x-auto">
              <code>{command.trim()}</code>
            </pre>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  handleRunCommand(command.trim());
                }}
                className="px-3 py-1 bg-orange-600 hover:bg-orange-500 text-white rounded text-xs transition-colors"
              >
                ✓ Run it
              </button>
              <button
                onClick={() => {}}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs transition-colors"
              >
                ✗ Skip
              </button>
            </div>
          </div>
          {after && <p className="whitespace-pre-wrap">{after.trim()}</p>}
        </div>
      );
    }

    // Inline code blocks (triple backtick without bash)
    const codeMatch = content.match(/```([\s\S]*?)```/);
    if (codeMatch) {
      const [fullMatch, code] = codeMatch;
      const before = content.slice(0, content.indexOf(fullMatch));
      const after = content.slice(content.indexOf(fullMatch) + fullMatch.length);
      return (
        <div className="relative z-10 space-y-2">
          {before && <p className="whitespace-pre-wrap">{before.trim()}</p>}
          <pre className="bg-gray-950/80 text-green-300 p-2 rounded text-xs font-mono overflow-x-auto">
            <code>{code.trim()}</code>
          </pre>
          {after && <p className="whitespace-pre-wrap">{after.trim()}</p>}
        </div>
      );
    }

    return <p className="relative z-10 whitespace-pre-wrap">{content}</p>;
  };

  return (
    /*
     * Root element fills the OS window (which Tauri has made transparent & frameless).
     * The flame animation border IS the window chrome.
     */
    <div
      className="w-screen h-screen flex flex-col overflow-hidden"
      style={{ background: 'transparent' }}
    >
      {/* Outer flame container — rounded corners create the "floating" look */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="relative flex flex-col w-full h-full rounded-2xl overflow-hidden"
        style={{
          background:
            'linear-gradient(180deg, rgba(20,5,0,0.97) 0%, rgba(10,3,0,0.98) 100%)',
          boxShadow:
            '0 0 60px rgba(255,80,0,0.5), 0 0 120px rgba(255,40,0,0.25), inset 0 0 40px rgba(255,69,0,0.08)',
          border: '1.5px solid rgba(255,100,0,0.35)',
        }}
      >
        {/* Flame + Ember background layers */}
        <FlameBackground />
        <EmberOverlay />

        {/* ── Title-bar drag region ── */}
        <div
          data-tauri-drag-region
          className="relative z-20 flex items-center justify-between px-3 pt-2 pb-1 select-none cursor-grab active:cursor-grabbing"
        >
          {/* Logo + title */}
          <div className="flex items-center gap-2" data-tauri-drag-region>
            <motion.img
              src={topLeftLogo}
              alt="Lilim"
              className="w-7 h-7 object-contain"
              animate={{
                filter: [
                  'drop-shadow(0 0 4px rgba(255,69,0,0.8))',
                  'drop-shadow(0 0 10px rgba(255,69,0,1))',
                  'drop-shadow(0 0 4px rgba(255,69,0,0.8))',
                ],
              }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            />
            <img src={bannerImage} alt="Lilith" className="h-6 object-contain opacity-90" />
          </div>

          {/* Window controls */}
          <div className="flex items-center gap-1">
            {isStreaming && (
              <motion.span
                className="text-orange-400 text-xs mr-2"
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.2, repeat: Infinity }}
              >
                🔥
              </motion.span>
            )}
            <button
              onClick={handleMinimize}
              className="w-6 h-6 rounded-full bg-yellow-500/80 hover:bg-yellow-400 flex items-center justify-center transition-colors group"
              title="Minimize"
            >
              <Minus size={10} className="text-yellow-900 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
            <button
              onClick={handleClose}
              className="w-6 h-6 rounded-full bg-red-500/80 hover:bg-red-400 flex items-center justify-center transition-colors group"
              title="Close"
            >
              <X size={10} className="text-red-900 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          </div>
        </div>

        {/* Thin glowing divider under title bar */}
        <div className="relative z-20 h-px mx-3 bg-gradient-to-r from-transparent via-orange-500/60 to-transparent" />

        {/* ── Message area ── */}
        <div className="relative flex-1 overflow-y-auto px-3 py-3 space-y-3 z-10 scrollbar-thin scrollbar-thumb-orange-800 scrollbar-track-transparent">
          {/* Faint center watermark */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none">
            <img
              src={centerLogo}
              alt=""
              className="w-48 h-48 object-contain opacity-[0.07]"
              style={{ filter: 'drop-shadow(0 0 20px rgba(255,69,0,0.3))' }}
            />
          </div>

          <AnimatePresence initial={false}>
            {messages.map(message => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 8, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.2 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[88%] px-3 py-2.5 rounded-2xl text-sm leading-relaxed relative overflow-hidden ${
                    message.role === 'user'
                      ? 'bg-gradient-to-br from-orange-600 to-red-700 text-white rounded-br-sm'
                      : 'bg-gray-900/80 text-gray-100 border border-orange-500/20 rounded-bl-sm'
                  }`}
                  style={
                    message.role === 'user'
                      ? {
                          boxShadow:
                            '0 0 20px rgba(255,100,0,0.35), inset 0 -1px 10px rgba(255,200,0,0.15)',
                        }
                      : {
                          boxShadow: '0 0 15px rgba(255,69,0,0.07)',
                        }
                  }
                >
                  {message.role !== 'user' && (
                    <div
                      className="absolute inset-0 opacity-10"
                      style={{
                        background:
                          'linear-gradient(to top, rgba(80,20,0,0) 0%, rgba(140,60,0,0.4) 100%)',
                      }}
                    />
                  )}
                  {renderContent(message)}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input bar ── */}
        <div className="relative z-20 px-3 pb-3 pt-2 border-t border-orange-500/20 bg-black/30">
          <div className="flex gap-2 items-end">
            <textarea
              rows={1}
              value={input}
              onChange={e => {
                setInput(e.target.value);
                // Auto-grow
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
              }}
              onKeyDown={handleKeyPress}
              placeholder={isStreaming ? 'Lilim is thinking...' : 'Ask anything...'}
              disabled={isStreaming}
              className="flex-1 resize-none bg-gray-900/70 text-white placeholder-gray-500 px-3 py-2 rounded-xl border border-orange-500/25 focus:border-orange-500/60 focus:outline-none focus:ring-1 focus:ring-orange-500/30 transition-all text-sm disabled:opacity-50 min-h-[38px] max-h-[120px] overflow-y-auto"
              style={{ boxShadow: 'inset 0 0 15px rgba(0,0,0,0.4)' }}
            />
            <motion.button
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.93 }}
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="w-10 h-10 flex-shrink-0 flex items-center justify-center bg-gradient-to-br from-orange-600 to-red-700 text-white rounded-xl hover:from-orange-500 hover:to-red-600 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ boxShadow: '0 0 15px rgba(255,80,0,0.4)' }}
            >
              <Send size={16} />
            </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}