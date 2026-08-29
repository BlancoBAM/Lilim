import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, X, Minus, Flame, Settings, User } from 'lucide-react';
import { FlameBackground } from './FlameBackground';
import { EmberOverlay } from './EmberOverlay';
import { SettingsPanel } from './SettingsPanel';
import bannerImage from '../assets/lilim-banner.svg';
import centerLogo from '../assets/03a17ee9fd4fe33c3ca16baf528b1598cfae5797.png';
import {
  streamChat, runShellCommand, sendObservation,
  getUserProfile, saveUserProfile,
  type LilimMessage
} from '../api/lilim';
import { getCurrentWindow } from '@tauri-apps/api/window';

const appWindow = getCurrentWindow();

import {
  GREETINGS,
  THINKING_MESSAGES,
  ERROR_MESSAGES
} from '../responses';

export function ChatInterface() {
  const [messages, setMessages] = useState<LilimMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: GREETINGS[Math.floor(Math.random() * GREETINGS.length)],
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [thinkingMsg, setThinkingMsg] = useState('');
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [showNoBrainBanner, setShowNoBrainBanner] = useState(false);
  // Pending confirmation state — set when server sends tool_pending
  const [pendingCommand, setPendingCommand] = useState<{ command: string; short: string } | null>(null);
  // First-launch profile modal
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [profileDraft, setProfileDraft] = useState({ display_name: '', github_username: '' });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Rotate thinking messages while streaming
  useEffect(() => {
    if (!isStreaming) { setThinkingMsg(''); return; }
    setThinkingMsg(THINKING_MESSAGES[Math.floor(Math.random() * THINKING_MESSAGES.length)]);
    const interval = setInterval(() => {
      setThinkingMsg(THINKING_MESSAGES[Math.floor(Math.random() * THINKING_MESSAGES.length)]);
    }, 3000);
    return () => clearInterval(interval);
  }, [isStreaming]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* ── Startup: health check + first-launch profile detection ── */
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8080/health');
        if (res.ok) {
          const data = await res.json();
          if ((data.providers_ready ?? 1) === 0) {
            setShowNoBrainBanner(true);
          }
        }
      } catch {
        // Backend not reachable — handled by streamChat error path
      }
    };

    const checkProfile = async () => {
      // Only show the profile modal once (stored in localStorage)
      const seen = localStorage.getItem('lilim_profile_seen');
      if (seen) return;
      const profile = await getUserProfile();
      if (profile) {
        setProfileDraft({
          display_name: profile.display_name || profile.system_username,
          github_username: profile.github_username || '',
        });
        // If no GitHub username is set yet, prompt the user
        if (!profile.github_username) {
          setShowProfileModal(true);
        } else {
          localStorage.setItem('lilim_profile_seen', '1');
        }
      }
    };

    checkHealth();
    checkProfile();
  }, []);

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
    const controller = new AbortController();
    setAbortController(controller);

    try {
      let accumulated = '';

      for await (const chunk of streamChat(userMessage.content, controller.signal)) {
        if (chunk.start) {
          setMessages(prev => [
            ...prev,
            { id: assistantId, role: 'assistant', content: '', timestamp: new Date() },
          ]);
          continue;
        }
        // @ts-ignore — check for pending command payload
        if (chunk.pending_command) {
          // Server wants confirmation before running this command
          setPendingCommand({
            // @ts-ignore
            command: chunk.pending_command,
            // @ts-ignore
            short: chunk.pending_short || chunk.pending_command.slice(0, 80),
          });
          continue;
        }
        if (chunk.end) {
          if (chunk.provider && chunk.provider !== 'PENDING') {
            setMessages(prev =>
              prev.map(m => (m.id === assistantId ? { ...m, provider: chunk.provider } : m))
            );
          }
          continue;
        }
        if (!chunk.content) continue;

        accumulated += chunk.content;
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, content: accumulated } : m))
        );
      }
    } catch (error) {
      if ((error as any).name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        setMessages(prev => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content:
              error instanceof Error
                ? `*${ERROR_MESSAGES[Math.floor(Math.random() * ERROR_MESSAGES.length)]} (${error.message})*`
                : `*${ERROR_MESSAGES[Math.floor(Math.random() * ERROR_MESSAGES.length)]}*`,
            timestamp: new Date(),
          },
        ]);
      }
    } finally {
      setIsStreaming(false);
      setAbortController(null);
    }
  }, [input, isStreaming]);

  const handleStop = useCallback(() => {
    if (abortController) {
      abortController.abort();
    }
  }, [abortController]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ── Shell command confirmation ── */
  const handleRunCommand = async (command: string) => {
    setPendingCommand(null);
    setIsStreaming(true);
    const execId = Date.now().toString();

    // Show a brief "executing..." indicator
    setMessages(prev => [
      ...prev,
      {
        id: execId,
        role: 'assistant',
        content: `*⚡ Running: \`${command.slice(0, 80)}\`...*`,
        timestamp: new Date(),
      },
    ]);

    try {
      const result = await runShellCommand(command);
      const stdout = (result.stdout || '').trim();
      const stderr = (result.stderr || '').trim();
      const output = [stdout, stderr].filter(Boolean).join('\n') || '(Command completed — no output)';
      const observationText = result.returncode === 0
        ? output
        : `Error (exit ${result.returncode}): ${output}`;

      // Remove the "executing" placeholder
      setMessages(prev => prev.filter(m => m.id !== execId));

      // Now stream the LLM's reaction to the real output
      const controller = new AbortController();
      setAbortController(controller);
      const obsId = (Date.now() + 1).toString();
      let accumulated = '';

      for await (const chunk of sendObservation(observationText, controller.signal)) {
        if (chunk.start) {
          setMessages(prev => [
            ...prev,
            { id: obsId, role: 'assistant', content: '', timestamp: new Date() },
          ]);
          continue;
        }
        if (chunk.end) continue;
        if (!chunk.content) continue;
        accumulated += chunk.content;
        setMessages(prev =>
          prev.map(m => (m.id === obsId ? { ...m, content: accumulated } : m))
        );
      }
    } catch (e) {
      setMessages(prev => prev.filter(m => m.id !== execId));
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `*Command failed: ${e instanceof Error ? e.message : String(e)}*`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsStreaming(false);
      setAbortController(null);
    }
  };

  /* ── Skip a pending command ── */
  const handleSkipCommand = async () => {
    setPendingCommand(null);
    setIsStreaming(true);
    const controller = new AbortController();
    setAbortController(controller);
    const skipId = Date.now().toString();
    let accumulated = '';

    try {
      for await (const chunk of sendObservation('[User declined to run the command. Do not retry it.]', controller.signal)) {
        if (chunk.start) {
          setMessages(prev => [
            ...prev,
            { id: skipId, role: 'assistant', content: '', timestamp: new Date() },
          ]);
          continue;
        }
        if (chunk.end) continue;
        if (!chunk.content) continue;
        accumulated += chunk.content;
        setMessages(prev =>
          prev.map(m => (m.id === skipId ? { ...m, content: accumulated } : m))
        );
      }
    } catch {
      // ignore
    } finally {
      setIsStreaming(false);
      setAbortController(null);
    }
  };

  /* ── Save first-launch profile ── */
  const handleSaveProfile = async () => {
    await saveUserProfile(profileDraft);
    localStorage.setItem('lilim_profile_seen', '1');
    setShowProfileModal(false);
  };

  /* ── Render a single message bubble ── */
  const renderContent = (message: LilimMessage) => {
    if (message.role === 'user') {
      return <p className="whitespace-pre-wrap relative z-10">{message.content}</p>;
    }

    const content = message.content;

    // Detect if ANY bash block in this message was already auto-executed by the
    // ReAct agent loop. The observation marker **[System →`...`]** is injected
    // immediately after each auto-executed block.
    const wasAutoExecuted = content.includes('**[System \u2192') || content.includes('**[System →');

    // Build rendered segments by parsing the content for code blocks
    const segments: React.ReactNode[] = [];
    let segIdx = 0;

    // Regex: captures optional lang tag + code body
    const CODE_BLOCK_RE = /```(bash|sh|shell)?\n?([\s\S]*?)```/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = CODE_BLOCK_RE.exec(content)) !== null) {
      const lang = (match[1] || '').toLowerCase();
      const code = match[2].trim();
      const isBash = lang === 'bash' || lang === 'sh' || lang === 'shell';

      // Text before this block
      if (match.index > lastIndex) {
        const before = content.slice(lastIndex, match.index);
        if (before.trim()) {
          segments.push(
            <p key={`t-${segIdx++}`} className="whitespace-pre-wrap">{before.trim()}</p>
          );
        }
      }

      if (isBash && !wasAutoExecuted) {
        // Bash block that hasn't been auto-executed yet → show confirmation UI
        segments.push(
          <div key={`bash-${segIdx++}`} className="bg-black/40 border border-orange-500/40 rounded-lg p-3">
            <p className="text-orange-300 text-xs mb-2 flex items-center gap-1">
              <Flame size={12} /> System command requested:
            </p>
            <pre className="bg-gray-950/80 text-green-300 p-2 rounded text-xs font-mono mb-3 overflow-x-auto">
              <code>{code}</code>
            </pre>
            <div className="flex gap-2">
              <button
                onClick={() => handleRunCommand(code)}
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
        );
      } else {
        // Auto-executed bash block OR plain code block → just show as code
        const label = isBash && wasAutoExecuted ? '▶ executed' : (lang || 'code');
        segments.push(
          <div key={`code-${segIdx++}`}>
            {isBash && wasAutoExecuted && (
              <p className="text-green-400/70 text-[10px] mb-1 font-mono">✓ {label}</p>
            )}
            <pre className="bg-gray-950/80 text-green-300 p-2 rounded text-xs font-mono overflow-x-auto">
              <code>{code}</code>
            </pre>
          </div>
        );
      }

      lastIndex = match.index + match[0].length;
    }

    // Remaining text after last code block
    if (lastIndex < content.length) {
      const tail = content.slice(lastIndex).trim();
      if (tail) {
        segments.push(
          <p key={`t-${segIdx++}`} className="whitespace-pre-wrap">{tail}</p>
        );
      }
    }

    if (segments.length === 0) {
      return <p className="relative z-10 whitespace-pre-wrap">{content}</p>;
    }

    return <div className="relative z-10 space-y-2">{segments}</div>;
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
          className="relative z-20 flex flex-col px-3 pt-2 pb-1 select-none"
        >
          {/* Top row: logo + window controls */}
          <div className="flex items-center justify-between cursor-grab active:cursor-grabbing" data-tauri-drag-region>
            {/* Banner — stretches to fill available header space */}
            <div className="flex-1 flex items-center" data-tauri-drag-region>
              <img
                src={bannerImage}
                alt="Lilim"
                className="w-full max-h-10 object-contain object-left opacity-95"
              />
            </div>

            {/* Window controls — icon-only, always visible */}
            <div className="flex items-center gap-2.5">
              {isStreaming && (
                <motion.span
                  className="text-orange-400 text-xs mr-1"
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 1.2, repeat: Infinity }}
                >
                  🔥
                </motion.span>
              )}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center justify-center text-blue-300/80 hover:text-blue-200 transition-colors"
                title="Settings"
              >
                <Settings size={13} />
              </button>
              <button
                onClick={handleMinimize}
                className="flex items-center justify-center text-amber-400/80 hover:text-amber-300 transition-colors"
                title="Minimize"
              >
                <Minus size={13} />
              </button>
              <button
                onClick={handleClose}
                className="flex items-center justify-center text-red-400/80 hover:text-red-300 transition-colors"
                title="Close"
              >
                <X size={13} />
              </button>
            </div>
          </div>

          {/* No-provider banner — shown when providers_ready is 0 */}
          {showNoBrainBanner && (
            <div className="mt-1.5 flex items-center gap-2 bg-amber-900/50 border border-amber-500/40 rounded-lg px-2 py-1.5 text-[10px] text-amber-200">
              <span className="flex-1">
                ⚠ No AI provider — local mode only. Add a key in{' '}
                <button
                  onClick={() => { setShowSettings(true); setShowNoBrainBanner(false); }}
                  className="underline text-amber-300 hover:text-amber-100"
                >
                  Settings
                </button>.
              </span>
              <button onClick={() => setShowNoBrainBanner(false)} className="text-amber-400 hover:text-amber-200 font-bold">✕</button>
            </div>
          )}
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
                    <>
                      <div
                        className="absolute inset-0 opacity-10"
                        style={{
                          background:
                            'linear-gradient(to top, rgba(80,20,0,0) 0%, rgba(140,60,0,0.4) 100%)',
                        }}
                      />
                      {message.provider && (
                        <div className="absolute top-1 right-2 text-[8px] font-bold text-orange-500/50 uppercase tracking-tighter pointer-events-none">
                          {message.provider}
                        </div>
                      )}
                    </>
                  )}
                  {renderContent(message)}

                  {/* Pending command confirmation card — inline at bottom of assistant message */}
                  {pendingCommand && message.id === messages[messages.length - 1]?.id && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-3 bg-black/50 border border-orange-500/50 rounded-lg p-3"
                    >
                      <p className="text-orange-300 text-xs mb-2 flex items-center gap-1">
                        <Flame size={12} /> Confirm command:
                      </p>
                      <pre className="bg-gray-950/80 text-green-300 p-2 rounded text-xs font-mono mb-3 overflow-x-auto whitespace-pre-wrap">
                        <code>{pendingCommand.command}</code>
                      </pre>
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleRunCommand(pendingCommand.command)}
                          className="px-3 py-1 bg-orange-600 hover:bg-orange-500 text-white rounded text-xs transition-colors"
                        >
                          ✓ Run it
                        </button>
                        <button
                          onClick={handleSkipCommand}
                          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs transition-colors"
                        >
                          ✗ Skip
                        </button>
                      </div>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        <div className="relative z-20 px-3 pb-3 pt-2 border-t border-orange-500/20 bg-black/30">
          <div className="flex gap-2 items-end">
            <div className="flex-1 flex flex-col gap-1">
              {isStreaming && (
                <motion.div 
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="px-2 text-[10px] text-white font-bold italic"
                >
                  {thinkingMsg}
                </motion.div>
              )}
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
                placeholder={isStreaming ? "" : 'Ask anything...'}
                disabled={isStreaming}
                className={`w-full resize-none bg-gray-900/70 text-white px-3 py-2.5 rounded-xl border border-orange-500/25 focus:border-orange-500/60 focus:outline-none focus:ring-1 focus:ring-orange-500/30 transition-all text-sm disabled:opacity-50 min-h-[42px] max-h-[120px] overflow-y-auto ${
                  isStreaming ? 'placeholder-transparent' : 'placeholder-gray-500'
                }`}
                style={{ boxShadow: 'inset 0 0 15px rgba(0,0,0,0.4)' }}
              />
            </div>
            <motion.button
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.93 }}
              onClick={isStreaming ? handleStop : handleSend}
              disabled={!isStreaming && !input.trim()}
              className={`w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-xl transition-all ${
                isStreaming 
                ? 'bg-red-600 hover:bg-red-500 text-white' 
                : 'bg-gradient-to-br from-orange-600 to-red-700 text-white hover:from-orange-500 hover:to-red-600'
              }`}
              style={{ boxShadow: isStreaming ? '0 0 15px rgba(255,0,0,0.4)' : '0 0 15px rgba(255,80,0,0.4)' }}
            >
              {isStreaming ? <X size={18} /> : <Send size={16} />}
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Settings Overlay */}
      <AnimatePresence>
        {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
      </AnimatePresence>

      {/* First-launch profile modal */}
      <AnimatePresence>
        {showProfileModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              className="w-80 bg-gray-950 border border-orange-500/40 rounded-2xl p-5 shadow-2xl"
              style={{ boxShadow: '0 0 40px rgba(255,80,0,0.25)' }}
            >
              <div className="flex items-center gap-2 mb-4">
                <User size={16} className="text-orange-400" />
                <h2 className="text-white text-sm font-semibold">Quick Setup</h2>
              </div>
              <p className="text-gray-400 text-xs mb-4 leading-relaxed">
                I'll work better knowing who I'm talking to. You can change these any time in Settings.
              </p>

              <label className="block text-gray-400 text-xs mb-1">Your name / username</label>
              <input
                type="text"
                value={profileDraft.display_name}
                onChange={e => setProfileDraft(d => ({ ...d, display_name: e.target.value }))}
                placeholder="e.g. alex"
                className="w-full bg-gray-900 text-white text-sm px-3 py-2 rounded-lg border border-orange-500/25 focus:border-orange-500/60 focus:outline-none mb-3"
              />

              <label className="block text-gray-400 text-xs mb-1">GitHub username <span className="text-gray-600">(for push/pull)</span></label>
              <input
                type="text"
                value={profileDraft.github_username}
                onChange={e => setProfileDraft(d => ({ ...d, github_username: e.target.value }))}
                placeholder="e.g. BlancoBAM"
                className="w-full bg-gray-900 text-white text-sm px-3 py-2 rounded-lg border border-orange-500/25 focus:border-orange-500/60 focus:outline-none mb-4"
              />

              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => { localStorage.setItem('lilim_profile_seen', '1'); setShowProfileModal(false); }}
                  className="px-3 py-1.5 text-gray-500 hover:text-gray-300 text-xs transition-colors"
                >
                  Skip
                </button>
                <button
                  onClick={handleSaveProfile}
                  className="px-4 py-1.5 bg-orange-600 hover:bg-orange-500 text-white text-xs rounded-lg transition-colors"
                >
                  Save
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}