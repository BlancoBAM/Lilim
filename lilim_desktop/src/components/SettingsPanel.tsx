import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { X, Save, Cpu, Cloud, Key, ChevronDown, ChevronUp, CheckCircle, AlertCircle, RefreshCw, Zap } from 'lucide-react';
import { getModelStatus, getProvidersStatus, registerApiKey, saveModelConfig, type ProviderStatus } from '../api/lilim';

// ── Provider metadata for display ────────────────────────────────────────────

const PROVIDER_DISPLAY: Record<string, {
  label: string;
  url: string;
  freeNote: string;
  color: string;
  keyPlaceholder: string;
  keyPrefix?: string;
}> = {
  openrouter: {
    label: 'OpenRouter',
    url: 'https://openrouter.ai',
    freeNote: '30+ free models via one key',
    color: '#8B5CF6',
    keyPlaceholder: 'sk-or-v1-…',
    keyPrefix: 'sk-or-v1-',
  },
  groq: {
    label: 'Groq',
    url: 'https://console.groq.com',
    freeNote: '14,400 req/day free — fastest inference',
    color: '#F59E0B',
    keyPlaceholder: 'gsk_…',
    keyPrefix: 'gsk_',
  },
  gemini: {
    label: 'Google Gemini',
    url: 'https://aistudio.google.com',
    freeNote: '500 req/day free',
    color: '#4285F4',
    keyPlaceholder: 'AIza…',
    keyPrefix: 'AIza',
  },
  cerebras: {
    label: 'Cerebras',
    url: 'https://cloud.cerebras.ai',
    freeNote: '14,400 req/day free — ultra fast',
    color: '#10B981',
    keyPlaceholder: 'csk-…',
    keyPrefix: 'csk-',
  },
  cloudflare: {
    label: 'Cloudflare AI',
    url: 'https://developers.cloudflare.com/workers-ai',
    freeNote: '10,000 neurons/day free',
    color: '#F97316',
    keyPlaceholder: 'CF token (also needs Account ID)',
  },
  cohere: {
    label: 'Cohere',
    url: 'https://cohere.com',
    freeNote: '1,000 req/month free',
    color: '#6366F1',
    keyPlaceholder: '40-char alphanumeric key',
  },
  mistral: {
    label: 'Mistral',
    url: 'https://console.mistral.ai',
    freeNote: 'Free experiment plan (needs phone verification)',
    color: '#EF4444',
    keyPlaceholder: 'Mistral API key',
  },
  huggingface: {
    label: 'HuggingFace',
    url: 'https://huggingface.co',
    freeNote: '$0.10/month credits — access to all open models',
    color: '#FBBF24',
    keyPlaceholder: 'hf_…',
    keyPrefix: 'hf_',
  },
  deepseek: {
    label: 'DeepSeek',
    url: 'https://platform.deepseek.com',
    freeNote: 'Generous free tier — strong reasoning',
    color: '#06B6D4',
    keyPlaceholder: 'dsk-…',
    keyPrefix: 'dsk-',
  },
  openai: {
    label: 'OpenAI',
    url: 'https://platform.openai.com',
    freeNote: 'Paid — $5 trial credits on signup',
    color: '#374151',
    keyPlaceholder: 'sk-…',
    keyPrefix: 'sk-',
  },
  anthropic: {
    label: 'Anthropic',
    url: 'https://console.anthropic.com',
    freeNote: 'Paid — no free tier',
    color: '#7C3AED',
    keyPlaceholder: 'sk-ant-…',
    keyPrefix: 'sk-ant-',
  },
};

const FREE_PROVIDERS = ['openrouter', 'groq', 'gemini', 'cerebras', 'cloudflare', 'cohere', 'mistral', 'huggingface', 'deepseek'];
const PAID_PROVIDERS = ['openai', 'anthropic'];

const STORAGE_KEY = 'lilim_model_config';

function loadConfig(): Record<string, string> {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  } catch { return {}; }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ title, icon, children, defaultOpen = true }: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-orange-500/20 rounded-xl overflow-hidden mb-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-orange-900/20 hover:bg-orange-900/30 transition-colors"
      >
        <span className="flex items-center gap-2 text-orange-200 text-sm font-medium">{icon}{title}</span>
        {open ? <ChevronUp size={14} className="text-orange-400" /> : <ChevronDown size={14} className="text-orange-400" />}
      </button>
      {open && <div className="px-3 pt-2 pb-3 space-y-3">{children}</div>}
    </div>
  );
}

function ProviderRow({
  providerId,
  status,
  savedKey,
  onSave,
}: {
  providerId: string;
  status?: ProviderStatus;
  savedKey: string;
  onSave: (key: string, provider: string) => void;
}) {
  const [keyInput, setKeyInput] = useState(savedKey ? '●●●●●●●●●●●●' : '');
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const display = PROVIDER_DISPLAY[providerId];
  const isConfigured = status?.configured ?? (savedKey.length > 0);

  const handleSave = async () => {
    if (!keyInput.trim() || keyInput === '●●●●●●●●●●●●') return;
    setSaving(true);
    const trimmedKey = keyInput.trim();
    try {
      // ALWAYS persist locally first — don't gate on backend response
      const cfg = loadConfig();
      cfg[`${providerId}Key`] = trimmedKey;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));

      // Try to register with backend (best-effort)
      try {
        await registerApiKey(trimmedKey, providerId);
        await saveModelConfig(cfg);
      } catch {
        // Backend call failed — local save already succeeded above
      }

      // Update parent state and show confirmation
      onSave(trimmedKey, providerId);
      setSaved(true);
      setEditing(false);
      setKeyInput('●●●●●●●●●●●●');
      // Confirmation persists — no auto-hide
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-lg border border-white/5 bg-black/20 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: display?.color ?? '#666', boxShadow: isConfigured ? `0 0 6px ${display?.color}` : 'none' }}
          />
          <span className="text-sm text-white font-medium">{display?.label ?? providerId}</span>
          {(isConfigured || saved) && !editing && (
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-1 text-xs text-green-400 font-semibold"
            >
              <CheckCircle size={12} /> {saved ? '✓ Saved!' : '✓ Configured'}
            </motion.span>
          )}
        </div>
        <a
          href={display?.url}
          target="_blank"
          rel="noreferrer"
          className="text-[10px] text-orange-400 hover:text-orange-300 underline transition-colors"
        >
          Get key ↗
        </a>
      </div>

      <p className="text-[10px] text-gray-500 mb-2">{display?.freeNote}</p>

      {status && (
        <div className="flex gap-3 mb-2 text-[10px] text-gray-600">
          <span>≤{status.daily_limit.toLocaleString()}/day</span>
          <span>{(status.tokens_per_min / 1000).toFixed(0)}k tok/min</span>
          {status.failures > 0 && <span className="text-red-400">{status.failures} recent failures</span>}
        </div>
      )}

      <div className="flex gap-1.5">
        {editing ? (
          <>
            <input
              type="password"
              value={keyInput}
              onChange={e => setKeyInput(e.target.value)}
              placeholder={display?.keyPlaceholder ?? 'API key'}
              className="flex-1 bg-black/40 text-white text-xs px-2 py-1.5 rounded-lg border border-orange-500/30 focus:border-orange-500/60 focus:outline-none placeholder-gray-600"
              autoFocus
              onKeyDown={e => e.key === 'Enter' && handleSave()}
            />
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-2 py-1.5 bg-green-700 hover:bg-green-600 text-white text-xs rounded-lg transition-colors disabled:opacity-50"
            >
              {saving ? <RefreshCw size={10} className="animate-spin" /> : <Save size={10} />}
            </button>
            <button
              onClick={() => { setEditing(false); setKeyInput(isConfigured ? '●●●●●●●●●●●●' : ''); }}
              className="px-2 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded-lg transition-colors"
            >
              <X size={10} />
            </button>
          </>
        ) : (
          <button
            onClick={() => { setEditing(true); setKeyInput(''); }}
            className={`flex-1 py-1.5 text-xs rounded-lg transition-colors ${
              isConfigured
                ? 'bg-green-900/30 text-green-300 hover:bg-green-900/50 border border-green-500/20'
                : 'bg-orange-900/20 text-orange-300 hover:bg-orange-900/40 border border-orange-500/20'
            }`}
          >
            {isConfigured ? '✓ Key configured — click to update' : '+ Enter API key'}
          </button>
        )}
      </div>
    </div>
  );
}

// ── Main Settings Panel ───────────────────────────────────────────────────────

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const [config, setConfig] = useState<Record<string, string>>(loadConfig);
  const [providerStatuses, setProviderStatuses] = useState<ProviderStatus[]>([]);
  const [modelStatus, setModelStatus] = useState<{
    available: boolean; device: string; source: string; size_mb: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [strategy, setStrategy] = useState<'local-first' | 'free-first' | 'quality-first'>(
    (config.strategy as 'local-first' | 'free-first' | 'quality-first') ?? 'local-first'
  );

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    try {
      const [providers, model, backendConfig] = await Promise.all([
        getProvidersStatus(),
        getModelStatus(),
        import('../api/lilim').then(api => api.getModelConfig()),
      ]);

      // Sync backend config to local state and localStorage
      if (backendConfig && Object.keys(backendConfig).length > 0) {
        const merged = { ...loadConfig(), ...backendConfig };
        setConfig(merged);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
        if (merged.strategy) setStrategy(merged.strategy as any);
      }

      if (providers) setProviderStatuses(providers.providers);
      if (model) {
        setModelStatus({
          available: model.local_engine.available,
          device: model.local_engine.device,
          source: model.local_engine.model_status.source,
          size_mb: model.local_engine.model_status.size_mb,
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleKeySaved = (key: string, provider: string) => {
    const updated = { ...config, [`${provider}Key`]: key };
    setConfig(updated);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    // Re-fetch provider statuses after a short delay
    setTimeout(fetchStatus, 1000);
  };

  const handleStrategySave = async () => {
    const updated = { ...config, strategy };
    setConfig(updated);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    await saveModelConfig(updated);
  };

  const configuredCount = providerStatuses.filter(p => p.configured).length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 flex flex-col rounded-2xl overflow-hidden"
      style={{ zIndex: 50, background: 'rgba(6,1,0,0.98)', border: '1.5px solid rgba(255,80,0,0.35)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-orange-500/20 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-orange-200 font-medium text-sm">⚙ Settings</span>
          {!loading && (
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${
              configuredCount > 0
                ? 'bg-green-900/40 text-green-400 border border-green-500/20'
                : 'bg-red-900/30 text-red-400 border border-red-500/20'
            }`}>
              {configuredCount > 0 ? `${configuredCount} provider${configuredCount > 1 ? 's' : ''} ready` : 'No providers configured'}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchStatus}
            className="text-gray-500 hover:text-orange-400 transition-colors"
            title="Refresh status"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div
        className="flex-1 overflow-y-auto px-3 pt-3 pb-4 min-h-0 space-y-1"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,80,0,0.3) transparent' }}
      >

        {/* Local Model Status */}
        <Section title="Local Model (Phi-2, Built-in)" icon={<Cpu size={13} />}>
          {loading ? (
            <div className="flex items-center gap-2 text-gray-500 text-xs py-2">
              <RefreshCw size={12} className="animate-spin" /> Checking status…
            </div>
          ) : modelStatus ? (
            <div className={`rounded-lg p-3 border text-sm ${
              modelStatus.available
                ? 'bg-green-900/20 border-green-500/20'
                : 'bg-orange-900/20 border-orange-500/20'
            }`}>
              <div className="flex items-center gap-2 mb-1">
                {modelStatus.available
                  ? <CheckCircle size={13} className="text-green-400" />
                  : <AlertCircle size={13} className="text-orange-400" />
                }
                <span className={`font-medium text-xs ${modelStatus.available ? 'text-green-300' : 'text-orange-300'}`}>
                  {modelStatus.available ? 'Phi-2 Ready ✓' : 'Phi-2 Not Available'}
                </span>
              </div>
              {modelStatus.available && (
                <div className="text-[10px] text-gray-500 space-y-0.5">
                  <div>Device: <span className="text-gray-400">{modelStatus.device}</span></div>
                  <div>Source: <span className="text-gray-400">{modelStatus.source}</span></div>
                  {modelStatus.size_mb > 0 && <div>Size: <span className="text-gray-400">{modelStatus.size_mb} MB</span></div>}
                </div>
              )}
              {!modelStatus.available && (
                <p className="text-[10px] text-gray-500 mt-1">
                  Model files not found. Online providers will be used instead.
                  Configure at least one provider below.
                </p>
              )}
            </div>
          ) : (
            <div className="text-xs text-gray-500 py-2 flex items-center gap-2">
              <AlertCircle size={12} className="text-orange-400" />
              Backend not reachable — is the service running?
            </div>
          )}

          <p className="text-[10px] text-gray-600 mt-1">
            Microsoft Phi-2 (2.7B) runs locally on CPU. No API key needed.
            Bundled in the Lilith Linux package.
          </p>
        </Section>

        {/* Routing Strategy */}
        <Section title="Routing Strategy" icon={<Zap size={13} />}>
          <div className="grid grid-cols-3 gap-1.5">
            {(['local-first', 'free-first', 'quality-first'] as const).map(s => (
              <button
                key={s}
                onClick={() => setStrategy(s)}
                className={`py-1.5 rounded-lg text-xs transition-colors ${
                  strategy === s ? 'bg-orange-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                {s === 'local-first' ? '🏠 Local first' : s === 'free-first' ? '🆓 Free first' : '⭐ Best quality'}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-gray-600">
            {strategy === 'local-first' && 'Uses Phi-2 for most queries, online for complex ones.'}
            {strategy === 'free-first' && 'Always tries free providers before paid ones. Good if local is slow.'}
            {strategy === 'quality-first' && 'Uses the best available model for every query.'}
          </p>
          <button
            onClick={handleStrategySave}
            className="w-full py-1.5 bg-orange-800/40 hover:bg-orange-800/60 text-orange-300 text-xs rounded-lg transition-colors"
          >
            Save Strategy
          </button>
        </Section>

        {/* Free API Providers */}
        <Section title="Free API Providers (priority order)" icon={<span className="text-xs">🆓</span>}>
          <p className="text-[10px] text-gray-500 mb-2">
            Add keys for any providers. Lilim auto-detects the provider from the key format.
            Free providers are tried first in the order shown. No key = skipped.
          </p>
          {FREE_PROVIDERS.map(pid => {
            const status = providerStatuses.find(p => p.name === pid);
            const savedKey = config[`${pid}Key`] ?? '';
            return (
              <ProviderRow
                key={pid}
                providerId={pid}
                status={status}
                savedKey={savedKey}
                onSave={handleKeySaved}
              />
            );
          })}
        </Section>

        {/* Paid Providers */}
        <Section title="Paid Providers (fallback)" icon={<Key size={13} />} defaultOpen={false}>
          <p className="text-[10px] text-gray-500 mb-2">
            Only used when free providers are exhausted or rate-limited.
          </p>
          {PAID_PROVIDERS.map(pid => {
            const status = providerStatuses.find(p => p.name === pid);
            const savedKey = config[`${pid}Key`] ?? '';
            return (
              <ProviderRow
                key={pid}
                providerId={pid}
                status={status}
                savedKey={savedKey}
                onSave={handleKeySaved}
              />
            );
          })}
        </Section>

        {/* Cloudflare Account ID (special case) */}
        {config['cloudflareKey'] && (
          <div className="rounded-lg border border-orange-500/10 bg-black/20 p-3">
            <p className="text-xs text-orange-300 mb-1">Cloudflare Account ID</p>
            <p className="text-[10px] text-gray-500 mb-2">Required alongside the Cloudflare API token.</p>
            <input
              type="text"
              placeholder="Your Cloudflare Account ID"
              defaultValue={config['cloudflareAccountId'] ?? ''}
              onChange={e => {
                const updated = { ...config, cloudflareAccountId: e.target.value };
                setConfig(updated);
                localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
              }}
              className="w-full bg-black/40 text-white text-xs px-2.5 py-1.5 rounded-lg border border-orange-500/20 focus:border-orange-500/50 focus:outline-none placeholder-gray-600"
            />
          </div>
        )}

        {/* Info box */}
        <div className="rounded-lg bg-blue-900/10 border border-blue-500/10 p-3">
          <p className="text-[10px] text-blue-300 font-medium mb-1">💡 Recommended free setup</p>
          <ol className="text-[10px] text-gray-500 space-y-1 list-decimal list-inside">
            <li>Sign up at <span className="text-orange-400">openrouter.ai</span> — one key, 30+ free models</li>
            <li>Sign up at <span className="text-orange-400">console.groq.com</span> — fastest free inference</li>
            <li>No credit card required for either</li>
          </ol>
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-3 py-2.5 border-t border-orange-500/20 bg-black/30 flex items-center justify-between">
        <span className="text-[10px] text-gray-600">
          Keys are stored locally. Never transmitted except to the selected provider.
        </span>
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-700/60 hover:bg-orange-600/60 text-white text-xs rounded-lg transition-colors"
        >
          <Cloud size={12} /> Done
        </button>
      </div>
    </motion.div>
  );
}
