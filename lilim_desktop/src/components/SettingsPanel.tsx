import { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Save, Cpu, Cloud, Key, ChevronDown, ChevronUp } from 'lucide-react';

interface ModelConfig {
  localEnabled: boolean;
  localEndpoint: string;   // Ollama or llama.cpp compatible endpoint
  localModel: string;

  // Remote providers
  openaiKey: string;
  openaiModel: string;

  anthropicKey: string;
  anthropicModel: string;

  googleKey: string;
  googleModel: string;

  groqKey: string;         // Free tier — Groq
  groqModel: string;

  openrouterKey: string;   // Free models via OpenRouter
  openrouterModel: string;

  customEndpoint: string;
  customKey: string;
  customModel: string;

  // Routing preference
  preferFree: boolean;
  strategy: 'local-first' | 'free-first' | 'quality-first';
}

const DEFAULTS: ModelConfig = {
  localEnabled: true,
  localEndpoint: 'http://127.0.0.1:11434',
  localModel: 'tinyllama:latest',

  openaiKey: '',
  openaiModel: 'gpt-4o-mini',

  anthropicKey: '',
  anthropicModel: 'claude-haiku-3-5',

  googleKey: '',
  googleModel: 'gemini-2.0-flash',

  groqKey: '',
  groqModel: 'llama3-8b-8192',

  openrouterKey: '',
  openrouterModel: 'meta-llama/llama-3-8b-instruct:free',

  customEndpoint: '',
  customKey: '',
  customModel: '',

  preferFree: true,
  strategy: 'local-first',
};

const STORAGE_KEY = 'lilim_model_config';

function loadConfig(): ModelConfig {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? { ...DEFAULTS, ...JSON.parse(saved) } : DEFAULTS;
  } catch { return DEFAULTS; }
}

function saveConfig(cfg: ModelConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
  // Also send to the backend so it takes effect immediately
  fetch('http://127.0.0.1:8080/settings/model-config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  }).catch(() => {}); // best-effort
}

// ── A collapsible section ─────────────────────────────────────
function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-orange-500/20 rounded-xl overflow-hidden mb-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-orange-900/20 hover:bg-orange-900/30 transition-colors"
      >
        <span className="flex items-center gap-2 text-orange-200 text-sm font-medium">{icon}{title}</span>
        {open ? <ChevronUp size={14} className="text-orange-400" /> : <ChevronDown size={14} className="text-orange-400" />}
      </button>
      {open && <div className="px-3 pt-2 pb-3 space-y-2">{children}</div>}
    </div>
  );
}

// ── Labelled text input ───────────────────────────────────────
function Field({ label, value, onChange, placeholder, type = 'text' }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string;
}) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:border-orange-500/50 focus:outline-none placeholder-gray-600"
      />
    </div>
  );
}

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const [cfg, setCfg] = useState<ModelConfig>(loadConfig);
  const [saved, setSaved] = useState(false);

  const set = <K extends keyof ModelConfig>(key: K, value: ModelConfig[K]) =>
    setCfg(prev => ({ ...prev, [key]: value }));

  const handleSave = () => {
    saveConfig(cfg);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 flex flex-col rounded-2xl overflow-hidden"
      style={{ zIndex: 50, background: 'rgba(8,2,0,0.97)', border: '1.5px solid rgba(255,80,0,0.35)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-orange-500/20 flex-shrink-0">
        <span className="text-orange-200 font-medium text-sm">⚙ Settings</span>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors"><X size={16} /></button>
      </div>

      {/* Body — scrollable */}
      <div className="flex-1 overflow-y-auto px-3 pt-3 pb-4 min-h-0"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,80,0,0.3) transparent' }}>

        {/* Strategy */}
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-2">Model Routing Strategy</label>
          <div className="grid grid-cols-3 gap-1.5">
            {(['local-first', 'free-first', 'quality-first'] as const).map(s => (
              <button key={s} onClick={() => set('strategy', s)}
                className={`py-1.5 rounded-lg text-xs transition-colors ${cfg.strategy === s ? 'bg-orange-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}>
                {s === 'local-first' ? '🏠 Local first' : s === 'free-first' ? '🆓 Free first' : '⭐ Best quality'}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-600 mt-1.5">
            {cfg.strategy === 'local-first' && 'Uses local model for everything, falls back to free tiers if needed.'}
            {cfg.strategy === 'free-first' && 'Tries free API tiers (Groq, OpenRouter, Gemini) before paid models.'}
            {cfg.strategy === 'quality-first' && 'Always uses the best available model for the task type.'}
          </p>
        </div>

        {/* Local model */}
        <Section title="Local Model (Ollama / llama.cpp)" icon={<Cpu size={13} />}>
          <div className="flex items-center gap-2 mb-2">
            <label className="text-xs text-gray-400">Enable local inference</label>
            <button onClick={() => set('localEnabled', !cfg.localEnabled)}
              className={`relative w-9 h-5 rounded-full transition-colors ${cfg.localEnabled ? 'bg-orange-600' : 'bg-gray-700'}`}>
              <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${cfg.localEnabled ? 'left-4.5' : 'left-0.5'}`} />
            </button>
          </div>
          <Field label="Endpoint" value={cfg.localEndpoint} onChange={v => set('localEndpoint', v)}
            placeholder="http://127.0.0.1:11434" />
          <div className="mt-2">
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select value={cfg.localModel} onChange={e => set('localModel', e.target.value)}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:outline-none mb-1">
              <option value="tinyllama:latest">TinyLlama 1.1B (tiny, fast)</option>
              <option value="qwen2.5:0.5b">Qwen2.5 0.5B (tiny)</option>
              <option value="phi3:mini">Phi-3 Mini 3.8B (balanced)</option>
              <option value="llama3.2:1b">Llama 3.2 1B (fast)</option>
              <option value="llama3.2:3b">Llama 3.2 3B (better)</option>
              <option value="mistral:7b">Mistral 7B (best local)</option>
              <option value="custom">Custom…</option>
            </select>
            {cfg.localModel === 'custom' && (
              <Field label="Custom model name" value={cfg.customModel} onChange={v => set('customModel', v)}
                placeholder="my-model:tag" />
            )}
            <p className="text-[10px] text-gray-600 mt-1">
              Install Ollama: <span className="text-orange-500 font-mono">curl https://ollama.ai/install.sh | sh</span><br />
              Pull a model: <span className="text-orange-500 font-mono">ollama pull tinyllama</span>
            </p>
          </div>
        </Section>

        {/* Free providers */}
        <Section title="Free API Tiers" icon={<span className="text-xs">🆓</span>}>
          <p className="text-[10px] text-gray-500 mb-2">These providers have generous free tiers — recommended for remote models.</p>

          <p className="text-xs text-orange-300 font-medium mt-1 mb-1">Groq (fastest free LLM API)</p>
          <Field label="API Key (groq.com — free signup)" value={cfg.groqKey} onChange={v => set('groqKey', v)}
            placeholder="gsk_…" type="password" />
          <div className="mt-1">
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select value={cfg.groqModel} onChange={e => set('groqModel', e.target.value)}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:outline-none">
              <option value="llama3-8b-8192">Llama 3 8B (free)</option>
              <option value="llama3-70b-8192">Llama 3 70B (free)</option>
              <option value="mixtral-8x7b-32768">Mixtral 8x7B (free)</option>
              <option value="gemma-7b-it">Gemma 7B (free)</option>
            </select>
          </div>

          <p className="text-xs text-orange-300 font-medium mt-3 mb-1">Google Gemini (free tier)</p>
          <Field label="API Key (aistudio.google.com — free)" value={cfg.googleKey} onChange={v => set('googleKey', v)}
            placeholder="AIza…" type="password" />
          <div className="mt-1">
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select value={cfg.googleModel} onChange={e => set('googleModel', e.target.value)}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:outline-none">
              <option value="gemini-2.0-flash">Gemini 2.0 Flash (free tier)</option>
              <option value="gemini-1.5-flash">Gemini 1.5 Flash (free tier)</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro (paid)</option>
            </select>
          </div>

          <p className="text-xs text-orange-300 font-medium mt-3 mb-1">OpenRouter (many free models)</p>
          <Field label="API Key (openrouter.ai — free models available)" value={cfg.openrouterKey}
            onChange={v => set('openrouterKey', v)} placeholder="sk-or-…" type="password" />
          <Field label="Model (use :free suffix for free tier)" value={cfg.openrouterModel}
            onChange={v => set('openrouterModel', v)} placeholder="meta-llama/llama-3-8b-instruct:free" />
        </Section>

        {/* Paid providers */}
        <Section title="Paid Providers (optional)" icon={<Key size={13} />}>
          <p className="text-xs text-orange-300 font-medium mt-1 mb-1">OpenAI</p>
          <Field label="API Key" value={cfg.openaiKey} onChange={v => set('openaiKey', v)} placeholder="sk-…" type="password" />
          <div className="mt-1">
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select value={cfg.openaiModel} onChange={e => set('openaiModel', e.target.value)}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:outline-none">
              <option value="gpt-4o-mini">GPT-4o Mini (cheapest)</option>
              <option value="gpt-4o">GPT-4o</option>
              <option value="o4-mini">o4-mini (reasoning)</option>
            </select>
          </div>

          <p className="text-xs text-orange-300 font-medium mt-3 mb-1">Anthropic</p>
          <Field label="API Key" value={cfg.anthropicKey} onChange={v => set('anthropicKey', v)} placeholder="sk-ant-…" type="password" />
          <div className="mt-1">
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select value={cfg.anthropicModel} onChange={e => set('anthropicModel', e.target.value)}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:outline-none">
              <option value="claude-haiku-3-5">Claude 3.5 Haiku (fastest/cheapest)</option>
              <option value="claude-sonnet-4-5">Claude Sonnet 4.5</option>
              <option value="claude-opus-4-5">Claude Opus 4.5</option>
            </select>
          </div>
        </Section>

        {/* Custom */}
        <Section title="Custom / Self-hosted Endpoint" icon={<Cloud size={13} />}>
          <p className="text-[10px] text-gray-500 mb-2">Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, etc.)</p>
          <Field label="Endpoint URL" value={cfg.customEndpoint} onChange={v => set('customEndpoint', v)}
            placeholder="http://192.168.1.100:8000/v1" />
          <Field label="API Key (if required)" value={cfg.customKey} onChange={v => set('customKey', v)}
            placeholder="optional" type="password" />
          <Field label="Model name" value={cfg.customModel} onChange={v => set('customModel', v)}
            placeholder="my-custom-model" />
        </Section>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-3 py-2.5 border-t border-orange-500/20 bg-black/30 flex items-center justify-between">
        <span className={`text-xs transition-colors ${saved ? 'text-green-400' : 'text-gray-600'}`}>
          {saved ? '✓ Saved!' : 'Changes require restart to take effect.'}
        </span>
        <button onClick={handleSave}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-600 hover:bg-orange-500 text-white text-xs rounded-lg transition-colors">
          <Save size={12} /> Save Settings
        </button>
      </div>
    </motion.div>
  );
}
