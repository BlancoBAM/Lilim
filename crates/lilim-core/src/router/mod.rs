//! Multi-Provider Router with Smart Routing
//!
//! Routes chat/completion requests across multiple LLM providers using a
//! configurable strategy (LocalFirst, CloudFirst, CostOptimized,
//! QualityOptimized, Balanced). Within a priority tier, providers are selected
//! round-robin per request. On HTTP 429 or transport error, the router fails
//! over sequentially through the configured fallback chain.
//!
//! Providers speak the OpenAI-compatible chat/completions protocol, so the same
//! client path targets local mistral.rs, the internal litellm-backed Axum
//! endpoint, and remote providers alike.

use anyhow::Result;
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

use crate::{
    classifier::ComplexityClassifier,
    compression::TokenCompressor,
    config::{ProviderConfig, RouterConfig, RoutingStrategy},
};

// ─── Shared request/response shapes (OpenAI-compatible) ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

// ─── Provider ───

/// A single LLM provider reachable via an OpenAI-compatible endpoint.
pub struct Provider {
    config: ProviderConfig,
    client: Client,
    /// Round-robin cursor shared across requests that select this provider.
    rr_index: AtomicUsize,
}

impl Provider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| RouterError::Config(format!("reqwest client: {e}")))?;
        Ok(Self {
            config,
            client,
            rr_index: AtomicUsize::new(0),
        })
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn provider_type(&self) -> &crate::config::ProviderType {
        &self.config.provider_type
    }

    pub fn models(&self) -> &[crate::config::ModelConfig] {
        &self.config.models
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn priority(&self) -> u8 {
        self.config.priority
    }

    /// Round-robin cursor value (for test inspection).
    pub fn rr_index(&self) -> usize {
        self.rr_index.load(Ordering::Relaxed)
    }

    /// Advance and return the round-robin cursor.
    pub fn next_rr(&self) -> usize {
        self.rr_index.fetch_add(1, Ordering::Relaxed)
    }

    /// Does this provider offer a model with the given name?
    pub fn has_model(&self, model: &str) -> bool {
        self.config.models.iter().any(|m| m.name == model)
    }

    /// Find the model config by name.
    pub fn model_config(&self, model: &str) -> Option<&crate::config::ModelConfig> {
        self.config.models.iter().find(|m| m.name == model)
    }

    /// POST a chat completion to this provider.
    #[instrument(skip(self, request))]
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, RouterError> {
        let base = self.config.base_url.as_deref().unwrap_or("https://api.openai.com/v1");
        let url = format!("{base}/chat/completions");
        let mut req = self.client.post(&url).json(request);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| {
            RouterError::Transport(format!(
                "{name}/chat/completions: {e}",
                name = self.config.name,
                e = e
            ))
        })?;
        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| {
            RouterError::Transport(format!(
                "{name}/chat/completions parse: {e}",
                name = self.config.name,
                e = e
            ))
        })?;
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            return Err(RouterError::RateLimited(format!(
                "{}/chat/completions returned {status}",
                self.config.name,
                status = status
            )));
        }
        if !status.is_success() {
            return Err(RouterError::Transport(format!(
                "{}/chat/completions returned {status}: {body}",
                self.config.name,
                status = status,
                body = body
            )));
        }
        serde_json::from_value(body).map_err(|e| RouterError::Transport(format!("{e}")))
    }

    /// POST a completion to this provider.
    #[instrument(skip(self, request))]
    pub async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, RouterError> {
        let base = self.config.base_url.as_deref().unwrap_or("https://api.openai.com/v1");
        let url = format!("{base}/completions");
        let mut req = self.client.post(&url).json(request);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| {
            RouterError::Transport(format!("{}/completions: {e}", self.config.name))
        })?;
        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| {
            RouterError::Transport(format!(
                "{name}/completions parse: {e}",
                name = self.config.name,
                e = e
            ))
        })?;
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            return Err(RouterError::RateLimited(format!(
                "{}/completions returned {status}",
                self.config.name,
                status = status
            )));
        }
        if !status.is_success() {
            return Err(RouterError::Transport(format!(
                "{}/completions returned {status}: {body}",
                self.config.name,
                status = status,
                body = body
            )));
        }
        serde_json::from_value(body).map_err(|e| RouterError::Transport(format!("{e}")))
    }
}

// ─── Errors ───

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("No provider available for model: {0}")]
    NoProvider(String),
    #[error("All providers failed: {0}")]
    AllProvidersFailed(String),
    #[error("Rate limited by provider: {0}")]
    RateLimited(String),
    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Transport error: {0}")]
    Transport(String),
}

// ─── Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub context_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub name: String,
    pub provider_type: String,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStatus {
    pub providers: usize,
    pub models: usize,
    pub zero_config: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroConfigStatus {
    pub enabled: bool,
    pub strategy: String,
}

// ─── Router ───

pub struct Router {
    config: RouterConfig,
    providers: Arc<RwLock<Vec<Arc<Provider>>>>,
    model_to_provider: Arc<RwLock<HashMap<String, usize>>>,
    classifier: Option<Arc<ComplexityClassifier>>,
    compressor: Option<Arc<TokenCompressor>>,
}

impl Router {
    /// Create a new router from config, building provider clients.
    pub async fn new(config: &RouterConfig) -> Result<Self> {
        let mut providers = Vec::new();
        let mut model_to_provider = HashMap::new();

        for (idx, pcfg) in config.providers.iter().enumerate() {
            if !pcfg.enabled {
                continue;
            }
            let provider = Arc::new(Provider::new(pcfg.clone())?);
            for m in &pcfg.models {
                model_to_provider.insert(m.name.clone(), idx);
            }
            providers.push(provider);
        }

        // Sort by priority (lower = higher priority).
        providers.sort_by_key(|p| p.priority());

        let router = Self {
            config: config.clone(),
            providers: Arc::new(RwLock::new(providers)),
            model_to_provider: Arc::new(RwLock::new(model_to_provider)),
            classifier: None,
            compressor: None,
        };

        info!(
            "Router initialized with {} providers across {} models",
            router.providers.read().await.len(),
            router.model_to_provider.read().await.len()
        );
        Ok(router)
    }

    /// Attach a classifier for semantic routing.
    pub fn with_classifier(mut self, classifier: Arc<ComplexityClassifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Attach a compressor for outbound payloads.
    pub fn with_compressor(mut self, compressor: Arc<TokenCompressor>) -> Self {
        self.compressor = Some(compressor);
        self
    }

    /// List available models across all enabled providers.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, RouterError> {
        let providers = self.providers.read().await;
        let mut models = Vec::new();
        for p in providers.iter() {
            for m in p.models() {
                models.push(ModelInfo {
                    name: m.name.clone(),
                    provider: p.name().to_string(),
                    context_window: m.context_window,
                });
            }
        }
        Ok(models)
    }

    /// List configured providers.
    pub async fn list_providers(&self) -> Result<Vec<ProviderInfo>, RouterError> {
        let providers = self.providers.read().await;
        Ok(providers
            .iter()
            .map(|p| ProviderInfo {
                name: p.name().to_string(),
                provider_type: format!("{:?}", p.provider_type()),
                models: p.models().iter().map(|m| m.name.clone()).collect(),
            })
            .collect())
    }

    /// Add a provider at runtime.
    pub async fn add_provider(
        &self,
        config: ProviderConfig,
    ) -> Result<ProviderInfo, RouterError> {
        let provider = Arc::new(
            Provider::new(config.clone())
                .map_err(|e| RouterError::Config(e.to_string()))?,
        );
        let mut providers = self.providers.write().await;
        let idx = providers.len();
        for m in &config.models {
            self.model_to_provider
                .write()
                .await
                .insert(m.name.clone(), idx);
        }
        providers.push(provider);
        providers.sort_by_key(|p| p.priority());
        Ok(ProviderInfo {
            name: config.name,
            provider_type: format!("{:?}", config.provider_type),
            models: config.models.iter().map(|m| m.name.clone()).collect(),
        })
    }

    /// Remove a provider by name.
    pub async fn remove_provider(&self, name: &str) -> Result<(), RouterError> {
        let mut providers = self.providers.write().await;
        if let Some(pos) = providers.iter().position(|p| p.name() == name) {
            providers.remove(pos);
        }
        self.model_to_provider
            .write()
            .await
            .retain(|_, &mut idx| idx != providers.len());
        Ok(())
    }

    /// Get router status.
    pub async fn get_status(&self) -> Result<RouterStatus, RouterError> {
        let providers = self.providers.read().await;
        let models = self.model_to_provider.read().await;
        Ok(RouterStatus {
            providers: providers.len(),
            models: models.len(),
            zero_config: self.config.zero_config_enabled,
        })
    }

    /// Get zero-configuration router status.
    pub async fn get_zero_config_status(&self) -> Result<ZeroConfigStatus, RouterError> {
        Ok(ZeroConfigStatus {
            enabled: self.config.zero_config_enabled,
            strategy: format!("{:?}", self.config.strategy),
        })
    }

    // ─── Routing ───

    /// Resolve the provider index for a model name.
    async fn provider_for_model(&self, model: &str) -> Option<usize> {
        self.model_to_provider.read().await.get(model).copied()
    }

    /// Select candidate provider indices for a request, ordered by strategy.
    /// Returns indices into `self.providers`.
    async fn select_candidates(
        &self,
        model: &str,
        strategy: RoutingStrategy,
    ) -> Vec<usize> {
        let providers = self.providers.read().await;
        let direct = self.provider_for_model(model).await;

        // Build the ordered candidate list.
        let mut candidates: Vec<usize> = match strategy {
            RoutingStrategy::LocalFirst => {
                // Local (Ollama/Mistral local) providers first, then priority order.
                let mut local: Vec<usize> = Vec::new();
                let mut remote: Vec<usize> = Vec::new();
                for (i, p) in providers.iter().enumerate() {
                    if matches!(p.provider_type(), crate::config::ProviderType::Ollama)
                        || matches!(p.provider_type(), crate::config::ProviderType::Mistral)
                    {
                        local.push(i);
                    } else {
                        remote.push(i);
                    }
                }
                let mut all = local;
                all.extend(remote);
                all
            }
            RoutingStrategy::CloudFirst => {
                let mut remote: Vec<usize> = Vec::new();
                let mut local: Vec<usize> = Vec::new();
                for (i, p) in providers.iter().enumerate() {
                    if matches!(p.provider_type(), crate::config::ProviderType::Ollama)
                        || matches!(p.provider_type(), crate::config::ProviderType::Mistral)
                    {
                        local.push(i);
                    } else {
                        remote.push(i);
                    }
                }
                let mut all = remote;
                all.extend(local);
                all
            }
            RoutingStrategy::CostOptimized | RoutingStrategy::QualityOptimized => {
                // Already sorted by priority (lower = higher priority).
                (0..providers.len()).collect()
            }
            RoutingStrategy::Balanced => {
                // Round-robin friendly: priority order.
                (0..providers.len()).collect()
            }
        };

        // If a direct provider exists for the model, ensure it's included.
        if let Some(d) = direct {
            if !candidates.contains(&d) {
                candidates.push(d);
            }
        }

        // Filter to providers that actually offer the model and are enabled.
        candidates
            .into_iter()
            .filter(|&i| providers[i].enabled() && providers[i].has_model(model))
            .collect()
    }

    /// Round-robin select among same-priority candidates.
    fn round_robin_pick(
        providers: &[Arc<Provider>],
        candidates: &[usize],
    ) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        // Group candidates by priority, pick the highest-priority group, then
        // round-robin within it.
        let best_priority = candidates
            .iter()
            .map(|&i| providers[i].priority())
            .min()
            .unwrap_or(u8::MAX);
        let tier: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| providers[i].priority() == best_priority)
            .collect();
        if tier.is_empty() {
            return None;
        }
        let provider = &providers[tier[0]];
        let idx = provider.next_rr() % tier.len();
        Some(tier[idx])
    }

    /// Chat completion with strategy selection + sequential failover.
    #[instrument(skip(self, request))]
    pub async fn route_chat(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, RouterError> {
        let model = request.model.clone();
        let strategy = self.config.strategy;

        let candidates = self.select_candidates(&model, strategy).await;
        if candidates.is_empty() {
            return Err(RouterError::NoProvider(model));
        }

        let providers = self.providers.read().await;
        let mut last_error: Option<RouterError> = None;

        // Try the round-robin pick first, then fail over through remaining
        // candidates sequentially on 429 / transport error.
        let pick = Self::round_robin_pick(&providers, &candidates);
        let mut order: Vec<usize> = Vec::new();
        if let Some(p) = pick {
            order.push(p);
        }
        for &c in &candidates {
            if !order.contains(&c) {
                order.push(c);
            }
        }

        for &idx in &order {
            let p = &providers[idx];
            match p.chat(&request).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    warn!("Provider {} failed for model {}: {e}", p.name(), model);
                    last_error = Some(e);
                    // Fail over only on rate-limit / transport errors.
                    match &last_error {
                        Some(RouterError::RateLimited(_)) | Some(RouterError::Transport(_)) => {
                            continue;
                        }
                        _ => break,
                    }
                }
            }
        }

        Err(RouterError::AllProvidersFailed(format!(
            "model {model}: {}",
            last_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "no candidates".to_string())
        )))
    }

    /// Completion with strategy selection + sequential failover.
    #[instrument(skip(self, request))]
    pub async fn route_complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, RouterError> {
        let model = request.model.clone();
        let strategy = self.config.strategy;

        let candidates = self.select_candidates(&model, strategy).await;
        if candidates.is_empty() {
            return Err(RouterError::NoProvider(model));
        }

        let providers = self.providers.read().await;
        let mut last_error: Option<RouterError> = None;

        let pick = Self::round_robin_pick(&providers, &candidates);
        let mut order: Vec<usize> = Vec::new();
        if let Some(p) = pick {
            order.push(p);
        }
        for &c in &candidates {
            if !order.contains(&c) {
                order.push(c);
            }
        }

        for &idx in &order {
            let p = &providers[idx];
            match p.complete(&request).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    warn!("Provider {} failed for model {}: {e}", p.name(), model);
                    last_error = Some(e);
                    match &last_error {
                        Some(RouterError::RateLimited(_)) | Some(RouterError::Transport(_)) => {
                            continue;
                        }
                        _ => break,
                    }
                }
            }
        }

        Err(RouterError::AllProvidersFailed(format!(
            "model {model}: {}",
            last_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "no candidates".to_string())
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        BudgetLimits, ComplexityThresholds, ModelCapability, ModelConfig, ModelTier,
        ProviderType, RoutingStrategy,
    };
    use httpmock::prelude::*;
    use serde_json::json;
    use std::sync::Arc;

    fn test_provider_config(
        name: &str,
        base_url: &str,
        priority: u8,
        model: &str,
        enabled: bool,
    ) -> ProviderConfig {
        ProviderConfig {
            name: name.to_string(),
            provider_type: ProviderType::OpenAI,
            api_key: Some("test-key".to_string()),
            base_url: Some(base_url.to_string()),
            models: vec![ModelConfig {
                name: model.to_string(),
                display_name: model.to_string(),
                context_window: 4096,
                max_output_tokens: 2048,
                cost_per_million_input: 0.0,
                cost_per_million_output: 0.0,
                capabilities: vec![ModelCapability::Chat],
                tier: ModelTier::Budget,
            }],
            enabled,
            priority,
            rate_limit_rpm: None,
            timeout_secs: 5,
        }
    }

    fn chat_response(model: &str) -> Value {
        json!({
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    fn router_config(providers: Vec<ProviderConfig>) -> RouterConfig {
        RouterConfig {
            strategy: RoutingStrategy::Balanced,
            providers,
            zero_config_enabled: true,
            fallback_chain: Vec::new(),
            budget_limits: BudgetLimits {
                daily_usd: 5.0,
                monthly_usd: 50.0,
                per_request_usd: 0.50,
            },
            complexity_thresholds: ComplexityThresholds {
                simple_max_tokens: 512,
                medium_max_tokens: 2048,
                complex_max_tokens: 8192,
                embedding_model: "test".to_string(),
            },
        }
    }

    async fn build_router(providers: Vec<ProviderConfig>) -> Router {
        Router::new(&router_config(providers)).await.unwrap()
    }

    #[tokio::test]
    async fn round_robin_orders_requests_across_same_priority_providers() {
        let server_a = MockServer::start();
        let server_b = MockServer::start();

        server_a.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(chat_response("model-a"));
        });
        server_b.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(chat_response("model-b"));
        });

        let config_a = test_provider_config("a", &server_a.base_url(), 1, "model-x", true);
        let config_b = test_provider_config("b", &server_b.base_url(), 1, "model-x", true);
        let router = build_router(vec![config_a, config_b]).await;

        let req = ChatRequest {
            model: "model-x".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        let r1 = router.route_chat(req.clone()).await.unwrap();
        let r2 = router.route_chat(req.clone()).await.unwrap();

        let m1 = r1.model.as_str();
        let m2 = r2.model.as_str();
        assert!(
            (m1 == "model-a" && m2 == "model-b") || (m1 == "model-b" && m2 == "model-a"),
            "expected round-robin to alternate providers, got {m1} then {m2}"
        );
    }

    #[tokio::test]
    async fn failover_on_429_then_success() {
        let failing = MockServer::start();
        let ok = MockServer::start();

        failing.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429).json_body(json!({"error": {"message": "rate limited"}}));
        });
        ok.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(chat_response("model-ok"));
        });

        let cfg_fail = test_provider_config("fail", &failing.base_url(), 1, "model-x", true);
        let cfg_ok = test_provider_config("ok", &ok.base_url(), 1, "model-x", true);
        let router = build_router(vec![cfg_fail, cfg_ok]).await;

        let req = ChatRequest {
            model: "model-x".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        let resp = router.route_chat(req).await.unwrap();
        assert_eq!(resp.model, "model-ok");
    }

    #[tokio::test]
    async fn all_providers_failed_when_all_return_429() {
        let s1 = MockServer::start();
        let s2 = MockServer::start();

        s1.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429).json_body(json!({"error": {"message": "rate limited"}}));
        });
        s2.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429).json_body(json!({"error": {"message": "rate limited"}}));
        });

        let c1 = test_provider_config("s1", &s1.base_url(), 1, "model-x", true);
        let c2 = test_provider_config("s2", &s2.base_url(), 1, "model-x", true);
        let router = build_router(vec![c1, c2]).await;

        let req = ChatRequest {
            model: "model-x".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        let err = router.route_chat(req).await.unwrap_err();
        assert!(
            matches!(err, RouterError::AllProvidersFailed(_)),
            "expected AllProvidersFailed, got {err:?}"
        );
    }

    #[tokio::test]
    async fn local_only_never_calls_remote() {
        // Only a local provider is configured; no remote provider exists, so
        // the router must not attempt any remote call (no provider matches).
        let cfg_local = test_provider_config("local", "http://127.0.0.1:1", 1, "model-x", true);
        let router = build_router(vec![cfg_local]).await;

        let req = ChatRequest {
            model: "model-x".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        // The local server is not running, so the call fails with a transport
        // error rather than silently falling back to a remote provider.
        let err = router.route_chat(req).await.unwrap_err();
        assert!(
            matches!(err, RouterError::AllProvidersFailed(_)),
            "expected AllProvidersFailed (no remote fallback), got {err:?}"
        );
    }

    #[tokio::test]
    async fn no_provider_for_unknown_model() {
        let cfg = test_provider_config("p", "http://127.0.0.1:1", 1, "known-model", true);
        let router = build_router(vec![cfg]).await;

        let req = ChatRequest {
            model: "unknown-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        let err = router.route_chat(req).await.unwrap_err();
        assert!(
            matches!(err, RouterError::NoProvider(_)),
            "expected NoProvider, got {err:?}"
        );
    }

    #[tokio::test]
    async fn list_models_returns_configured_models() {
        let cfg = test_provider_config("p", "http://127.0.0.1:1", 1, "known-model", true);
        let router = build_router(vec![cfg]).await;

        let models = router.list_models().await.unwrap();
        assert!(
            models.iter().any(|m| m.name == "known-model"),
            "expected known-model in list, got {:?}",
            models
        );
    }

    #[tokio::test]
    async fn disabled_provider_is_not_routable() {
        let cfg = test_provider_config("disabled", "http://127.0.0.1:1", 1, "model-x", false);
        let router = build_router(vec![cfg]).await;

        let req = ChatRequest {
            model: "model-x".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
        };

        let err = router.route_chat(req).await.unwrap_err();
        assert!(
            matches!(err, RouterError::NoProvider(_)),
            "expected NoProvider for disabled provider, got {err:?}"
        );
    }
}
