import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, X, Flame, Settings } from 'lucide-react';
import bannerImage from '../assets/c80b4d356e3c7b98f2baabf558ea7bacc2421ec9.png';
import centerLogo from '../assets/03a17ee9fd4fe33c3ca16baf528b1598cfae5797.png';
import topLeftLogo from '../assets/51350c1f0fe5a2742ba35cd8899037600d9d9f62.png';
import { streamChat, runShellCommand, type LilimMessage } from '../api/lilim';
import { SettingsPanel } from './SettingsPanel';
import { getCurrentWindow } from '@tauri-apps/api/window';

const appWindow = getCurrentWindow();

// ── Ember particle canvas ─────────────────────────────────────
function EmberCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const resize = () => {
      const p = canvas.parentElement;
      if (p) { canvas.width = p.offsetWidth; canvas.height = p.offsetHeight; }
    };
    resize();
    window.addEventListener('resize', resize);

    type Ember = { x: number; y: number; vx: number; vy: number; size: number; life: number; hue: number };
    const embers: Ember[] = [];
    const spawn = () => ({
      x: Math.random() * canvas.width,
      y: canvas.height + 5,
      vx: (Math.random() - 0.5) * 2,
      vy: -(Math.random() * 2 + 0.8),
      size: Math.random() * 2 + 0.5,
      life: 1,
      hue: Math.random() * 40 + 5,
    });
    for (let i = 0; i < 15; i++) { const e = spawn(); e.y = Math.random() * canvas.height; embers.push(e); }

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (embers.length < 20 && Math.random() < 0.07) embers.push(spawn());
      for (let i = embers.length - 1; i >= 0; i--) {
        const e = embers[i];
        e.x += e.vx + Math.sin(e.life * 10) * 0.5;
        e.y += e.vy;
        e.vx += (Math.random() - 0.5) * 0.15;
        e.life -= 0.006;
        if (e.life <= 0 || e.y < -10) { embers.splice(i, 1); continue; }
        const g = ctx.createRadialGradient(e.x, e.y, 0, e.x, e.y, e.size * 3);
        g.addColorStop(0, `hsla(${e.hue},100%,65%,${e.life * 0.9})`);
        g.addColorStop(1, `hsla(${e.hue},100%,40%,0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(e.x, e.y, e.size * 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = `hsla(${e.hue + 30},100%,90%,${e.life})`;
        ctx.beginPath(); ctx.arc(e.x, e.y, e.size * 0.6, 0, Math.PI * 2); ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(raf); };
  }, []);
  return <canvas ref={ref} className="absolute inset-0 pointer-events-none" style={{ zIndex: 1, mixBlendMode: 'screen' }} />;
}

// ── Flame edge glow ───────────────────────────────────────────
function FlameEdge() {
  return (
    <div className="absolute inset-0 pointer-events-none rounded-2xl overflow-hidden" style={{ zIndex: 0 }}>
      {/* Bottom flame gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32"
        style={{ background: 'linear-gradient(to top, rgba(255,80,0,0.35) 0%, rgba(255,60,0,0.18) 40%, transparent 100%)' }} />
      {/* Left/right edge glow */}
      <div className="absolute top-0 bottom-0 left-0 w-1"
        style={{ background: 'linear-gradient(to right, rgba(255,80,0,0.6), transparent)' }} />
      <div className="absolute top-0 bottom-0 right-0 w-1"
        style={{ background: 'linear-gradient(to left, rgba(255,80,0,0.6), transparent)' }} />
    </div>
  );
}

export function ChatInterface() {
  const [messages, setMessages] = useState<LilimMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Greetings, seeker. I am Lilim, your guide through the flames of knowledge. What wisdom do you seek today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleClose = () => appWindow.close();
  const handleMinimize = () => appWindow.minimize();

  const handleSend = useCallback(async () => {
    if (!input.trim() || isStreaming) return;
    const userMsg: LilimMessage = { id: Date.now().toString(), role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    if (textareaRef.current) { textareaRef.current.style.height = '38px'; }
    setIsStreaming(true);

    const asstId = (Date.now() + 1).toString();
    let accumulated = '';

    try {
      for await (const chunk of streamChat(userMsg.content)) {
        if (chunk.start) {
          setMessages(prev => [...prev, { id: asstId, role: 'assistant', content: '', timestamp: new Date() }]);
          continue;
        }
        if (chunk.end || !chunk.content) continue;
        accumulated += chunk.content;
        setMessages(prev => prev.map(m => m.id === asstId ? { ...m, content: accumulated } : m));
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(), role: 'assistant', timestamp: new Date(),
        content: err instanceof Error ? `*${err.message}*` : '*An unknown error occurred.*',
      }]);
    } finally {
      setIsStreaming(false);
    }
  }, [input, isStreaming]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleRunCommand = async (command: string) => {
    try {
      const result = await runShellCommand(command);
      const out = (result.stdout || result.stderr || '(no output)').trim();
      setMessages(prev => [...prev, {
        id: Date.now().toString(), role: 'assistant', timestamp: new Date(),
        content: `\`\`\`\n${out}\n\`\`\``,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(), role: 'assistant', timestamp: new Date(),
        content: `*Command failed: ${e instanceof Error ? e.message : String(e)}*`,
      }]);
    }
  };

  const renderContent = (msg: LilimMessage) => {
    if (msg.role === 'user') return <p className="whitespace-pre-wrap text-sm">{msg.content}</p>;

    // bash block → confirmation UI
    const bashMatch = msg.content.match(/```bash\n?([\s\S]*?)```/);
    if (bashMatch) {
      const cmd = bashMatch[1].trim();
      const before = msg.content.slice(0, msg.content.indexOf(bashMatch[0])).trim();
      const after = msg.content.slice(msg.content.indexOf(bashMatch[0]) + bashMatch[0].length).trim();
      return (
        <div className="space-y-2 text-sm">
          {before && <p className="whitespace-pre-wrap">{before}</p>}
          <div className="bg-black/50 border border-orange-500/40 rounded-lg p-2.5">
            <p className="text-orange-300 text-xs mb-1.5 flex items-center gap-1"><Flame size={10} /> Command requested:</p>
            <pre className="bg-gray-950/90 text-green-300 p-2 rounded text-xs font-mono mb-2 overflow-x-auto whitespace-pre-wrap"><code>{cmd}</code></pre>
            <div className="flex gap-2">
              <button onClick={() => handleRunCommand(cmd)}
                className="px-2.5 py-1 bg-orange-600 hover:bg-orange-500 text-white rounded text-xs transition-colors">✓ Run</button>
              <button className="px-2.5 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs transition-colors">✗ Skip</button>
            </div>
          </div>
          {after && <p className="whitespace-pre-wrap">{after}</p>}
        </div>
      );
    }

    // generic code block
    const codeMatch = msg.content.match(/```(?:\w+)?\n?([\s\S]*?)```/);
    if (codeMatch) {
      const code = codeMatch[1].trim();
      const before = msg.content.slice(0, msg.content.indexOf(codeMatch[0])).trim();
      const after = msg.content.slice(msg.content.indexOf(codeMatch[0]) + codeMatch[0].length).trim();
      return (
        <div className="space-y-2 text-sm">
          {before && <p className="whitespace-pre-wrap">{before}</p>}
          <pre className="bg-gray-950/90 text-green-300 p-2 rounded text-xs font-mono overflow-x-auto"><code>{code}</code></pre>
          {after && <p className="whitespace-pre-wrap">{after}</p>}
        </div>
      );
    }

    return <p className="whitespace-pre-wrap text-sm">{msg.content}</p>;
  };

  return (
    <div className="w-screen h-screen" style={{ background: 'transparent' }}>
      {/* Outer glow wrapper — this IS the visible "window" */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.25 }}
        className="w-full h-full flex flex-col rounded-2xl overflow-hidden relative"
        style={{
          background: 'linear-gradient(180deg, #160500 0%, #0e0300 50%, #080200 100%)',
          border: '1.5px solid rgba(255,90,0,0.4)',
          boxShadow: '0 0 50px rgba(255,70,0,0.45), 0 0 100px rgba(255,40,0,0.2)',
        }}
      >
        {/* Decorative layers */}
        <FlameEdge />
        <EmberCanvas />

        {/* ── Title bar (drag region) ── */}
        <div
          data-tauri-drag-region
          className="relative flex items-center justify-between px-3 py-2 flex-shrink-0 select-none"
          style={{ zIndex: 10, background: 'linear-gradient(180deg, rgba(255,60,0,0.15) 0%, transparent 100%)' }}
        >
          <div className="flex items-center gap-2" data-tauri-drag-region>
            <motion.img src={topLeftLogo} alt="" className="w-6 h-6 object-contain"
              animate={{ filter: ['drop-shadow(0 0 3px rgba(255,80,0,0.8))', 'drop-shadow(0 0 8px rgba(255,80,0,1))', 'drop-shadow(0 0 3px rgba(255,80,0,0.8))'] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            />
            <img src={bannerImage} alt="Lilith" className="h-5 object-contain opacity-90" />
          </div>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <motion.span className="text-orange-400 text-[10px]"
                animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity }}>
                🔥 thinking…
              </motion.span>
            )}
            <button onClick={() => setShowSettings(true)}
              className="w-5 h-5 flex items-center justify-center text-orange-400/60 hover:text-orange-300 transition-colors" title="Settings">
              <Settings size={13} />
            </button>
            <button onClick={handleMinimize}
              className="w-5 h-5 rounded-full bg-yellow-500/70 hover:bg-yellow-400 transition-colors" title="Minimize" />
            <button onClick={handleClose}
              className="w-5 h-5 rounded-full bg-red-500/70 hover:bg-red-400 transition-colors group flex items-center justify-center" title="Close">
              <X size={9} className="text-red-900 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          </div>
        </div>

        {/* Divider */}
        <div className="flex-shrink-0 h-px mx-3" style={{ background: 'linear-gradient(to right, transparent, rgba(255,100,0,0.5), transparent)', zIndex: 10 }} />

        {/* ── Messages ── */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2 min-h-0"
          style={{ zIndex: 10, scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,80,0,0.3) transparent' }}>
          {/* Faint center watermark */}
          <div className="pointer-events-none select-none absolute inset-0 flex items-center justify-center" style={{ zIndex: 0 }}>
            <img src={centerLogo} alt="" className="w-32 h-32 object-contain"
              style={{ opacity: 0.06, filter: 'drop-shadow(0 0 15px rgba(255,69,0,0.3))' }} />
          </div>

          <AnimatePresence initial={false}>
            {messages.map(msg => (
              <motion.div key={msg.id}
                initial={{ opacity: 0, y: 6, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.18 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                style={{ position: 'relative', zIndex: 5 }}
              >
                <div className={`max-w-[88%] px-3 py-2 rounded-2xl relative overflow-hidden ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-br from-orange-600 to-red-700 text-white rounded-br-sm'
                    : 'text-gray-100 rounded-bl-sm'
                }`}
                  style={msg.role === 'user'
                    ? { boxShadow: '0 0 18px rgba(255,100,0,0.3)' }
                    : { background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,80,0,0.15)', backdropFilter: 'blur(4px)' }
                  }
                >
                  {renderContent(msg)}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input bar ── */}
        <div className="flex-shrink-0 px-3 pb-3 pt-2 flex gap-2 items-end"
          style={{ zIndex: 10, borderTop: '1px solid rgba(255,80,0,0.15)', background: 'rgba(0,0,0,0.3)', backdropFilter: 'blur(4px)' }}>
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={e => {
              setInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
            }}
            onKeyDown={handleKeyPress}
            placeholder={isStreaming ? 'Lilim is thinking…' : 'Ask anything…'}
            disabled={isStreaming}
            className="flex-1 resize-none text-white placeholder-gray-600 px-3 py-2 rounded-xl text-sm disabled:opacity-50"
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,80,0,0.2)',
              outline: 'none',
              minHeight: '38px',
              maxHeight: '100px',
              boxShadow: 'inset 0 0 10px rgba(0,0,0,0.4)',
              color: 'white',
            }}
            onFocus={e => { e.target.style.borderColor = 'rgba(255,100,0,0.5)'; }}
            onBlur={e => { e.target.style.borderColor = 'rgba(255,80,0,0.2)'; }}
          />
          <motion.button
            whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.93 }}
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-xl text-white disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: 'linear-gradient(135deg, #ea580c, #b91c1c)', boxShadow: '0 0 12px rgba(255,80,0,0.4)' }}
          >
            <Send size={15} />
          </motion.button>
        </div>
      </motion.div>

      {/* Settings slide-in panel */}
      <AnimatePresence>
        {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
      </AnimatePresence>
    </div>
  );
}