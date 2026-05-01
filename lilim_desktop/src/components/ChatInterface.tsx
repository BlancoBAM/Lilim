import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Send, Maximize2, Minimize2 } from 'lucide-react';
import { FlameBackground } from './FlameBackground';
import { EmberOverlay } from './EmberOverlay';
import bannerImage from '../assets/c80b4d356e3c7b98f2baabf558ea7bacc2421ec9.png';
import centerLogo from '../assets/03a17ee9fd4fe33c3ca16baf528b1598cfae5797.png';
import topLeftLogo from '../assets/51350c1f0fe5a2742ba35cd8899037600d9d9f62.png';
import { streamChat, type LilimMessage } from '../api/lilim';

export function ChatInterface() {
  const [messages, setMessages] = useState<LilimMessage[]>([
    {
      id: '1',
      role: 'assistant',
      type: 'message',
      content: 'Greetings, seeker. I am Lilim, your guide through the flames of knowledge. What wisdom do you seek today?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isMaximized, setIsMaximized] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: LilimMessage = {
      id: Date.now().toString(),
      role: 'user',
      type: 'message',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsStreaming(true);

    try {
      // Track in-flight message assembly
      let currentAssistantId = (Date.now() + 1).toString();
      let currentContent = '';
      let currentType: LilimMessage['type'] = 'message';
      let currentFormat: string | undefined;
      let hasStarted = false;

      for await (const chunk of streamChat(userMessage.content)) {
        // Skip ephemeral chunks (active_line markers)
        if (chunk.format === 'active_line') continue;
        if (chunk.type === 'review') continue;

        // Handle new message type starting
        if (chunk.start) {
          // Save previous message if any
          if (hasStarted && currentContent) {
            setMessages(prev => {
              const withoutCurrent = prev.filter(m => m.id !== currentAssistantId);
              return [...withoutCurrent, {
                id: currentAssistantId,
                role: chunk.role === 'computer' ? 'computer' as const : 'assistant' as const,
                type: currentType,
                format: currentFormat,
                content: currentContent,
                timestamp: new Date(),
              }];
            });
          }

          // Start a new message
          currentAssistantId = (Date.now() + Math.random()).toString();
          currentContent = '';
          currentType = chunk.type as LilimMessage['type'];
          currentFormat = chunk.format;
          hasStarted = true;
          continue;
        }

        // Handle message type ending
        if (chunk.end) {
          if (currentContent) {
            setMessages(prev => {
              const withoutCurrent = prev.filter(m => m.id !== currentAssistantId);
              return [...withoutCurrent, {
                id: currentAssistantId,
                role: chunk.role === 'computer' ? 'computer' as const : 'assistant' as const,
                type: currentType,
                format: currentFormat,
                content: currentContent,
                timestamp: new Date(),
              }];
            });
          }
          hasStarted = false;
          continue;
        }

        // Handle confirmation (code execution approval)
        if (chunk.type === 'confirmation') {
          setMessages(prev => [...prev, {
            id: (Date.now() + Math.random()).toString(),
            role: 'assistant',
            type: 'confirmation',
            content: chunk.content,
            timestamp: new Date(),
          }]);
          continue;
        }

        // Accumulate content
        if (chunk.content) {
          currentContent += chunk.content;
          currentType = chunk.type as LilimMessage['type'];
          if (chunk.format) currentFormat = chunk.format;

          // Live-update the current message
          const liveId = currentAssistantId;
          const liveContent = currentContent;
          const liveType = currentType;
          const liveFormat = currentFormat;

          setMessages(prev => {
            const existing = prev.find(m => m.id === liveId);
            if (existing) {
              return prev.map(m =>
                m.id === liveId ? { ...m, content: liveContent } : m
              );
            } else {
              return [...prev, {
                id: liveId,
                role: chunk.role === 'computer' ? 'computer' as const : 'assistant' as const,
                type: liveType,
                format: liveFormat,
                content: liveContent,
                timestamp: new Date(),
              }];
            }
          });
        }
      }
    } catch (error) {
      console.error('Lilim Error:', error);
      const errorMessage: LilimMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        type: 'message',
        content: error instanceof Error
          ? `*The flames flicker... ${error.message}*`
          : '*The flames flicker... An unknown error occurred.*',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
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

  /** Render a message bubble based on its type */
  const renderMessageContent = (message: LilimMessage) => {
    // If it's a regular message, check for markdown code blocks, specifically ````bash`
    if (message.type === 'message' && message.role !== 'user') {
      const content = message.content;
      if (content.includes('```bash')) {
        const parts = content.split('```bash');
        const before = parts[0];
        const afterPart = parts[1];
        const commandEndIdx = afterPart.indexOf('```');
        
        if (commandEndIdx !== -1) {
          const command = afterPart.substring(0, commandEndIdx).trim();
          const after = afterPart.substring(commandEndIdx + 3);
          
          return (
            <div className="relative z-10">
              <p className="whitespace-pre-wrap mb-2">{before}</p>
              
              <div className="bg-orange-900/30 border border-orange-500/40 rounded-lg p-3 my-2">
                <p className="text-orange-200 text-sm mb-2">⚠️ System Command Requested:</p>
                <pre className="bg-gray-900/90 text-green-300 p-2 rounded text-xs font-mono mb-3 overflow-x-auto">
                  <code>{command}</code>
                </pre>
                <button
                  onClick={async () => {
                    try {
                      await fetch('http://127.0.0.1:8080/tools/shell', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ command, confirmed: true }),
                      });
                      alert('Command executed.');
                    } catch (e) {
                      alert('Execution failed.');
                    }
                  }}
                  className="px-4 py-1.5 bg-orange-600 hover:bg-orange-500 text-white rounded text-xs transition-colors"
                >
                  Run it
                </button>
              </div>

              <p className="whitespace-pre-wrap">{after}</p>
            </div>
          );
        }
      }
    }

    // Regular text message
    return <p className="relative z-10 whitespace-pre-wrap">{message.content}</p>;
  };

  /** Get bubble style based on message role and type */
  const getBubbleStyle = (message: LilimMessage, index: number) => {
    if (message.role === 'user') {
      return index % 2 === 0
        ? 'bg-gradient-to-br from-orange-600 via-orange-500 to-yellow-600 text-white'
        : 'bg-gradient-to-br from-red-600 via-orange-600 to-red-500 text-white';
    }

    if (message.type === 'code' || message.type === 'console') {
      return 'bg-transparent text-gray-100';
    }

    return index % 2 === 0
      ? 'bg-gradient-to-t from-gray-800/90 via-gray-700/80 to-gray-600/70 text-gray-100 border border-orange-500/20'
      : 'bg-gradient-to-t from-gray-900/90 via-gray-800/80 to-gray-700/70 text-gray-100 border border-red-500/20';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative bg-gradient-to-b from-orange-950/60 via-red-950/50 to-gray-900/80 rounded-lg shadow-2xl border border-orange-500/30 overflow-hidden h-full flex flex-col ${isMaximized ? 'fixed inset-4 z-50' : 'relative'
        }`}
      style={{
        boxShadow: '0 0 40px rgba(255, 69, 0, 0.2), inset 0 0 40px rgba(255, 69, 0, 0.05)',
      }}
    >
      {/* Flame Background - Bottom Layer */}
      <FlameBackground />

      {/* Ember Overlay - Top Layer */}
      <EmberOverlay />

      {/* Header */}
      <div className="relative bg-gradient-to-r from-orange-600 via-red-600 to-orange-600 p-6 border-b border-orange-500/50 z-10">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxkZWZzPjxwYXR0ZXJuIGlkPSJmbGFtZSIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiPjxwYXRoIGQ9Ik0gMCw1MCBRIDI1LDIwIDUwLDUwIFQgMTAwLDUwIiBzdHJva2U9InJnYmEoMjU1LDEyMCwwLDAuMykiIGZpbGw9Im5vbmUiIHN0cm9rZS13aWR0aD0iMiIvPjwvcGF0dGVybj48L2RlZnM+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0idXJsKCNmbGFtZSkiLz48L3N2Zz4=')] opacity-30" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-4">
            <motion.img
              src={topLeftLogo}
              alt="Lilim Logo"
              className="w-16 h-16 object-contain"
              animate={{
                filter: [
                  'drop-shadow(0 0 8px rgba(255, 69, 0, 0.8))',
                  'drop-shadow(0 0 16px rgba(255, 69, 0, 1))',
                  'drop-shadow(0 0 8px rgba(255, 69, 0, 0.8))',
                ],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
            <img
              src={bannerImage}
              alt="Lilith"
              className="h-16 object-contain"
            />
          </div>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <motion.div
                className="text-orange-300 text-xs"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                🔥 thinking...
              </motion.div>
            )}
            <button
              onClick={() => setIsMaximized(!isMaximized)}
              className="p-2 hover:bg-white/10 rounded transition-colors"
            >
              {isMaximized ? (
                <Minimize2 className="w-5 h-5 text-white" />
              ) : (
                <Maximize2 className="w-5 h-5 text-white" />
              )}
            </button>
          </div>
        </div>

        {/* Divider Line */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-orange-400 to-transparent" />
      </div>

      {/* Chat Messages */}
      <div className="relative flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar z-10">
        {/* Center Background Logo */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
          <img
            src={centerLogo}
            alt="Lilim Background"
            className="w-80 h-80 object-contain opacity-15"
            style={{
              filter: 'drop-shadow(0 0 20px rgba(255, 69, 0, 0.3))',
            }}
          />
        </div>

        {messages.map((message, index) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, x: message.role === 'user' ? 20 : -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} relative z-10`}
          >
            <div
              className={`max-w-[85%] px-4 py-3 rounded-lg relative overflow-hidden ${getBubbleStyle(message, index)}`}
              style={
                message.role === 'user'
                  ? {
                    boxShadow: '0 0 25px rgba(255, 120, 0, 0.4), inset 0 -2px 15px rgba(255, 200, 0, 0.2)',
                  }
                  : message.type === 'code' || message.type === 'console'
                    ? {}
                    : {
                      boxShadow: '0 0 20px rgba(255, 69, 0, 0.1)',
                    }
              }
            >
              {/* Smoke effect for AI messages */}
              {message.role !== 'user' && message.type === 'message' && (
                <div
                  className="absolute inset-0 opacity-20"
                  style={{
                    background: 'linear-gradient(to top, rgba(100, 100, 100, 0) 0%, rgba(150, 150, 150, 0.3) 100%)',
                  }}
                />
              )}
              {renderMessageContent(message)}
            </div>
          </motion.div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="relative p-4 bg-gray-900/80 border-t border-orange-500/30 z-10">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={isStreaming ? 'Lilim is thinking...' : 'Enter your query...'}
            disabled={isStreaming}
            className="flex-1 bg-gray-800/80 text-white px-4 py-3 rounded-lg border border-orange-500/30 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 transition-all placeholder-gray-500 disabled:opacity-50"
            style={{
              boxShadow: 'inset 0 0 20px rgba(0, 0, 0, 0.3)',
            }}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={isStreaming}
            className="px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-lg hover:from-orange-500 hover:to-red-500 transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              boxShadow: '0 0 20px rgba(255, 69, 0, 0.4)',
            }}
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}