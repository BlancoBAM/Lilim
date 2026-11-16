//! Unified API Gateway - Axum-based HTTP server replacing FastAPI

use axum::{
    extract::{Json, Path, Query, State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use rig::completion::{Message as RigMessage, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, instrument, warn};
use validator::Validate;

use crate::{
    agent::Agent,
    config::{Config, ModelCapability, ModelConfig, ModelTier, ProviderConfig, ProviderType},
    memory::Memory,
    mcp::McpHost,
    router::Router as ModelRouter,
    scheduler::Scheduler,
};

// ─── Shared state for the gateway ───

#[derive(Clone)]
pub struct GatewayState {
    pub agent: Arc<Agent>,
    pub memory: Arc<Memory>,
    pub router: Arc<ModelRouter>,
    pub mcp: Arc<McpHost>,
    pub scheduler: Arc<Scheduler>,
    pub config: Arc<Config>,
    pub ws_tx: broadcast::Sender<WsMessage>,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    ChatToken { token: String, done: bool },
    ToolCall { name: String, args: Value },
    ToolResult { name: String, result: Value, is_error: bool },
    Status { message: String },
    Error { message: String },
    Ping,
    Pong,
}

/// Gateway HTTP server
pub struct Gateway {
    state: GatewayState,
    router: Router,
}

impl Gateway {
    /// Create a new gateway with all routes configured
    pub async fn new(
        agent: Arc<Agent>,
        memory: Arc<Memory>,
        router: Arc<ModelRouter>,
        mcp: Arc<McpHost>,
        scheduler: Arc<Scheduler>,
        config: &Config,
    ) -> anyhow::Result<Self> {
        let (ws_tx, _) = broadcast::channel(1000);

        let state = GatewayState {
            agent,
            memory,
            router,
            mcp,
            scheduler,
            config: Arc::new(config.clone()),
            ws_tx,
        };

        let app = Router::new()
            // Health & info
            .route("/health", get(health))
            .route("/version", get(version))
            .route("/config", get(get_config))
            .route("/config", post(update_config))

            // Chat / Completions
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/models", get(list_models))
            .route("/chat/reset", post(chat_reset))

            // Agent loop
            .route("/agent/run", post(agent_run))
            .route("/agent/stream", get(agent_stream))

            // Memory
            .route("/memory/search", post(memory_search))
            .route("/memory/store", post(memory_store))
            .route("/memory/context", get(memory_context))
            .route("/memory/stats", get(memory_stats))
            .route("/memory/conversations", get(list_conversations))
            .route("/memory/conversations/:id", get(get_conversation))
            .route("/memory/conversations/:id", axum::routing::delete(delete_conversation))

            // MCP
            .route("/mcp/servers", get(list_mcp_servers))
            .route("/mcp/servers", post(add_mcp_server))
            .route("/mcp/servers/:name", axum::routing::delete(remove_mcp_server))
            .route("/mcp/tools", get(list_mcp_tools))
            .route("/mcp/call", post(call_mcp_tool))

            // Scheduler
            .route("/scheduler/schedule", post(schedule_task))
            .route("/scheduler/schedule/recurring", post(schedule_recurring))
            .route("/scheduler/list", get(list_schedules))
            .route("/scheduler/cancel/:id", post(cancel_schedule))

            // Routing / Providers
            .route("/router/providers", get(list_providers))
            .route("/router/providers", post(add_provider))
            .route("/router/providers/:name", axum::routing::delete(remove_provider))
            .route("/router/providers/context-limits", get(providers_context_limits))
            .route("/router/status", get(router_status))
            .route("/router/zero-config", get(zero_config_status))

            // WebSocket for real-time streaming
            .route("/ws", get(websocket_handler))

            .with_state(state.clone());

        Ok(Self {
            state,
            router: app,
        })
    }

    /// Get the Axum router
    pub fn router(&self) -> Router {
        self.router.clone()
    }

    /// Run the HTTP server
    pub async fn serve(self, addr: std::net::SocketAddr) -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("Gateway listening on {}", addr);
        axum::serve(listener, self.router).await?;
        Ok(())
    }
}

// ── Handlers ──

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "service": "lilim-core"}))
}

async fn version() -> impl IntoResponse {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "name": "lilim-core",
        "rust_version": option_env!("RUSTC_VERSION").unwrap_or("unknown"),
    }))
}

async fn get_config(State(state): State<GatewayState>) -> impl IntoResponse {
    // Return safe config (no API keys)
    let mut config = (*state.config).clone();
    // Redact sensitive fields
    for provider in &mut config.router.providers {
        provider.api_key = Some("***".to_string());
    }
    if let Some(key) = &mut config.memory.qdrant.api_key {
        *key = "***".to_string();
    }
    Json(config)
}

async fn update_config(
    State(_state): State<GatewayState>,
    Json(_payload): Json<Value>,
) -> impl IntoResponse {
    warn!("Config update requested");
    Json(serde_json::json!({"status": "config_update_not_implemented"}))
}

/// Chat completions endpoint (OpenAI-compatible)
#[tracing::instrument(skip(state, request))]
async fn chat_completions(
    State(state): State<GatewayState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, GatewayError> {
    let stream = request.stream.unwrap_or(false);

    if stream {
        let (tx, rx) = broadcast::channel(100);
        let agent = state.agent.clone();
        let messages: Vec<RigMessage> = request
            .messages
            .iter()
            .filter_map(|m| match m.role.as_str() {
                "user" => Some(RigMessage::user(m.content.clone())),
                "assistant" => Some(RigMessage::assistant(m.content.clone())),
                _ => None,
            })
            .collect();
        let model = request.model.clone();
        let max_tokens = request.max_tokens.unwrap_or(4096);
        let temperature = request.temperature.unwrap_or(0.7);

        tokio::spawn(async move {
            if let Err(e) = agent.stream_chat(messages, &model, Some(max_tokens), Some(temperature), tx).await {
                error!("Stream chat error: {}", e);
            }
        });

        let stream = tokio_stream::wrappers::BroadcastStream::new(rx);
        let body = axum::body::Body::from_stream(stream.map(|chunk| {
            let data = serde_json::to_string(&chunk.unwrap_or_default()).unwrap_or_default();
            Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", data))
        }));

        return Ok(Response::builder()
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap());
    }

    // Non-streaming: route through the router to a provider, then convert to
    // OpenAI-compatible chat response.
    let router_resp = state
        .router
        .route_chat(crate::router::ChatRequest {
            model: request.model,
            messages: request
                .messages
                .into_iter()
                .map(|m| crate::router::ChatMessage {
                    role: m.role,
                    content: m.content,
                    name: m.name,
                    tool_calls: m.tool_calls,
                    tool_call_id: m.tool_call_id,
                })
                .collect(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: None,
            stop: request.stop,
        })
        .await?;

    let response = ChatCompletionResponse {
        id: router_resp.id,
        object: "chat.completion".to_string(),
        created: router_resp.created,
        model: router_resp.model,
        choices: router_resp
            .choices
            .into_iter()
            .map(|c| ChatChoice {
                index: c.index,
                message: ChatMessage {
                    role: c.message.role,
                    content: c.message.content,
                    name: c.message.name,
                    tool_calls: c.message.tool_calls,
                    tool_call_id: c.message.tool_call_id,
                },
                finish_reason: c.finish_reason,
            })
            .collect(),
        usage: Usage {
            prompt_tokens: router_resp.usage.prompt_tokens,
            completion_tokens: router_resp.usage.completion_tokens,
            total_tokens: router_resp.usage.total_tokens,
        },
    };
    Ok(Json(response).into_response())
}

/// Legacy completions endpoint
async fn completions(
    State(state): State<GatewayState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, GatewayError> {
    let router_resp = state
        .router
        .route_complete(crate::router::CompletionRequest {
            model: request.model,
            prompt: request.prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: None,
        })
        .await?;

    let response = CompletionResponse {
        id: router_resp.id,
        object: "text_completion".to_string(),
        created: router_resp.created,
        model: router_resp.model,
        choices: router_resp
            .choices
            .into_iter()
            .map(|c| CompletionChoice {
                text: c.text,
                index: c.index,
                finish_reason: c.finish_reason,
            })
            .collect(),
        usage: Usage {
            prompt_tokens: router_resp.usage.prompt_tokens,
            completion_tokens: router_resp.usage.completion_tokens,
            total_tokens: router_resp.usage.total_tokens,
        },
    };
    Ok(Json(response))
}

async fn list_models(State(state): State<GatewayState>) -> Result<Json<ModelsResponse>, GatewayError> {
    let models = state
        .router
        .list_models()
        .await?
        .into_iter()
        .map(|m| ModelInfo {
            id: m.name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: m.provider,
            context_window: m.context_window,
            max_output_tokens: 4096,
            capabilities: vec!["chat".to_string()],
            tier: "standard".to_string(),
        })
        .collect();
    Ok(Json(ModelsResponse { models }))
}

/// Reset a chat session (clear history)
async fn chat_reset(
    State(state): State<GatewayState>,
    Json(request): Json<SessionResetRequest>,
) -> Result<Json<Value>, GatewayError> {
    state.memory.delete_conversation(&request.session_id).await?;
    Ok(Json(serde_json::json!({
        "status": "cleared",
        "session_id": request.session_id,
    })))
}

/// Get memory context for a query
async fn memory_context(
    State(state): State<GatewayState>,
    Query(params): Query<MemoryContextQuery>,
) -> Result<Json<Value>, GatewayError> {
    let context = state.memory.load_context(&params.query).await?;
    Ok(Json(serde_json::json!({ "context": context })))
}

/// Get memory statistics
async fn memory_stats(State(state): State<GatewayState>) -> Result<Json<MemoryStatsResponse>, GatewayError> {
    let stats = state.memory.stats().await?;
    Ok(Json(MemoryStatsResponse {
        memory_count: stats.memory_count,
        conversation_count: stats.conversation_count,
        message_count: stats.message_count,
    }))
}

/// Get provider context limits
async fn providers_context_limits(State(_state): State<GatewayState>) -> Result<Json<Value>, GatewayError> {
    Ok(Json(serde_json::json!({
        "providers": [],
        "safe_cap": 6144,
        "effective_cap": 6144,
        "user_override": false,
    })))
}

async fn agent_run(
    State(state): State<GatewayState>,
    Json(request): Json<AgentRunRequest>,
) -> Result<Json<AgentRunResponse>, GatewayError> {
    // Use the last user message as the prompt; chat history is passed through.
    let prompt = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let chat_history: Vec<RigMessage> = request
        .messages
        .iter()
        .filter_map(|m| match m.role.as_str() {
            "user" => Some(RigMessage::user(m.content.clone())),
            "assistant" => Some(RigMessage::assistant(m.content.clone())),
            _ => None,
        })
        .collect();

    let response_text = state.agent.run(&prompt, chat_history).await?;
    Ok(Json(AgentRunResponse {
        messages: vec![ChatMessage {
            role: "assistant".to_string(),
            content: response_text,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tool_calls: Vec::new(),
        iterations: 1,
        total_tokens: 0,
    }))
}

async fn agent_stream(
    State(state): State<GatewayState>,
    Query(params): Query<AgentStreamQuery>,
) -> impl IntoResponse {
    let (tx, rx) = broadcast::channel(100);
    let agent = state.agent.clone();
    let messages: Vec<RigMessage> = params
        .messages
        .iter()
        .filter_map(|m| match m.role.as_str() {
            "user" => Some(RigMessage::user(m.content.clone())),
            "assistant" => Some(RigMessage::assistant(m.content.clone())),
            _ => None,
        })
        .collect();
    let max_iterations = params.max_iterations.unwrap_or(10);

    tokio::spawn(async move {
        if let Err(e) = agent.stream_run(messages, vec![], Some(max_iterations), tx).await {
            error!("Agent stream error: {}", e);
        }
    });

    let stream = tokio_stream::wrappers::BroadcastStream::new(rx);
    let body = axum::body::Body::from_stream(stream.map(|chunk| {
        let data = serde_json::to_string(&chunk.unwrap_or_default()).unwrap_or_default();
        Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", data))
    }));

    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body)
        .unwrap()
}

/// Search memory (semantic + keyword)
#[tracing::instrument(skip(state))]
async fn memory_search(
    State(state): State<GatewayState>,
    Json(request): Json<MemorySearchRequest>,
) -> Result<Json<MemorySearchResponse>, GatewayError> {
    let results = state
        .memory
        .search(&request.query, request.limit, request.threshold)
        .await?
        .into_iter()
        .map(|r| MemorySearchResult {
            id: r.id,
            content: r.content,
            score: r.score,
            metadata: r.metadata,
            created_at: r.created_at,
        })
        .collect();
    Ok(Json(MemorySearchResponse { results }))
}

/// Store memory
#[tracing::instrument(skip(state))]
async fn memory_store(
    State(state): State<GatewayState>,
    Json(request): Json<MemoryStoreRequest>,
) -> Result<Json<MemoryStoreResponse>, GatewayError> {
    let id = state.memory.store(&request.content, request.metadata).await?;
    Ok(Json(MemoryStoreResponse { id }))
}

/// List conversations
async fn list_conversations(
    State(state): State<GatewayState>,
    Query(params): Query<ListConversationsQuery>,
) -> Result<Json<ConversationsResponse>, GatewayError> {
    let conversations = state
        .memory
        .list_conversations(params.limit, params.offset)
        .await?
        .into_iter()
        .map(|c| Conversation {
            id: c.id,
            title: c.title,
            messages: c
                .messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                    name: m.name,
                    tool_calls: m.tool_calls.map(|tc| tc.into_iter().map(|t| serde_json::json!({"id": t.id, "name": t.name, "arguments": t.arguments})).collect()),
                    tool_call_id: m.tool_call_id,
                })
                .collect(),
            created_at: c.created_at,
            updated_at: c.updated_at,
            token_count: c.token_count,
        })
        .collect();
    Ok(Json(ConversationsResponse { conversations }))
}

/// Get conversation by ID
async fn get_conversation(
    State(state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<Conversation>, GatewayError> {
    let conversation = state.memory.get_conversation(&id).await?;
    Ok(Json(Conversation {
        id: conversation.id,
        title: conversation.title,
        messages: conversation
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                name: m.name,
                tool_calls: m
                    .tool_calls
                    .map(|tc| {
                        tc.into_iter()
                            .map(|t| serde_json::json!({"id": t.id, "name": t.name, "arguments": t.arguments}))
                            .collect()
                    }),
                tool_call_id: m.tool_call_id,
            })
            .collect(),
        created_at: conversation.created_at,
        updated_at: conversation.updated_at,
        token_count: conversation.token_count,
    }))
}

/// Delete conversation
async fn delete_conversation(
    State(state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<DeleteResponse>, GatewayError> {
    state.memory.delete_conversation(&id).await?;
    Ok(Json(DeleteResponse { success: true }))
}

/// List MCP servers
async fn list_mcp_servers(State(state): State<GatewayState>) -> Result<Json<McpServersResponse>, GatewayError> {
    let servers = state
        .mcp
        .list_servers()
        .await?
        .into_iter()
        .map(|s| McpServerInfo {
            name: s.name,
            enabled: s.enabled,
            transport: s.transport,
            url: s.url,
            command: s.command,
            tools_count: s.tools_count,
        })
        .collect();
    Ok(Json(McpServersResponse { servers }))
}

/// Add MCP server
async fn add_mcp_server(
    State(state): State<GatewayState>,
    Json(request): Json<AddMcpServerRequest>,
) -> Result<Json<McpServerResponse>, GatewayError> {
    let server = state.mcp.add_server(request.name, request.config).await?;
    Ok(Json(McpServerResponse {
        server: McpServerInfo {
            name: server.name,
            enabled: server.enabled,
            transport: server.transport,
            url: server.url,
            command: server.command,
            tools_count: server.tools_count,
        },
    }))
}

/// Remove MCP server
async fn remove_mcp_server(
    State(state): State<GatewayState>,
    Path(name): Path<String>,
) -> Result<Json<DeleteResponse>, GatewayError> {
    state.mcp.remove_server(&name).await?;
    Ok(Json(DeleteResponse { success: true }))
}

/// List MCP tools
async fn list_mcp_tools(State(state): State<GatewayState>) -> Result<Json<McpToolsResponse>, GatewayError> {
    let tools = state
        .mcp
        .list_tools()
        .await?
        .into_iter()
        .map(|t| McpToolInfo {
            server: t.server,
            name: t.name,
            description: t.description,
            input_schema: t.input_schema,
            read_only: t.read_only,
        })
        .collect();
    Ok(Json(McpToolsResponse { tools }))
}

/// Call MCP tool
async fn call_mcp_tool(
    State(state): State<GatewayState>,
    Json(request): Json<CallMcpToolRequest>,
) -> Result<Json<CallMcpToolResponse>, GatewayError> {
    let result = state.mcp.call_tool(&request.server, &request.tool, request.arguments).await?;
    Ok(Json(CallMcpToolResponse { result }))
}

/// Schedule one-time task
async fn schedule_task(
    State(state): State<GatewayState>,
    Json(request): Json<ScheduleRequest>,
) -> Result<Json<ScheduleResponse>, GatewayError> {
    let result = state.scheduler.schedule_once(&request.message, &request.natural_time).await?;
    Ok(Json(ScheduleResponse {
        id: result["id"].as_i64().unwrap_or(0),
        message: request.message,
        trigger: request.natural_time,
        backend: result["backend"].as_str().unwrap_or("stub").to_string(),
        error: result["error"].as_str().map(|s| s.to_string()),
    }))
}

/// Schedule recurring task
async fn schedule_recurring(
    State(state): State<GatewayState>,
    Json(request): Json<ScheduleRecurringRequest>,
) -> Result<Json<ScheduleResponse>, GatewayError> {
    let result = state.scheduler.schedule_recurring(&request.message, &request.natural_time).await?;
    Ok(Json(ScheduleResponse {
        id: result["id"].as_i64().unwrap_or(0),
        message: request.message,
        trigger: request.natural_time,
        backend: result["backend"].as_str().unwrap_or("stub").to_string(),
        error: result["error"].as_str().map(|s| s.to_string()),
    }))
}

/// List schedules
 async fn list_schedules(
    State(state): State<GatewayState>,
    Query(params): Query<ListSchedulesQuery>,
) -> Result<Json<SchedulesResponse>, GatewayError> {
    let schedules = state
        .scheduler
        .list(params.active_only.unwrap_or(false))
        .await?
        .into_iter()
        .map(|s| ScheduleInfo {
            id: s.id,
            message: s.message,
            trigger: s.trigger,
            schedule_type: s.schedule_type,
            backend: s.backend,
            created_at: s.created_at,
            active: s.active,
        })
        .collect();
    Ok(Json(SchedulesResponse { schedules }))
}

/// Cancel schedule
async fn cancel_schedule(
    State(state): State<GatewayState>,
    Path(id): Path<i64>,
) -> Result<Json<ScheduleCancelResponse>, GatewayError> {
    let result = state.scheduler.cancel(id).await?;
    Ok(Json(ScheduleCancelResponse {
        id: result["id"].as_i64().unwrap_or(id),
        cancelled: result["cancelled"].as_bool().unwrap_or(false),
    }))
}

/// List providers
async fn list_providers(State(state): State<GatewayState>) -> Result<Json<ProvidersResponse>, GatewayError> {
    let providers = state
        .router
        .list_providers()
        .await?
        .into_iter()
        .map(|p| {
            let provider_name = p.name.clone();
            ProviderInfo {
                name: p.name,
                provider_type: p.provider_type,
                enabled: true,
                models: p
                    .models
                    .into_iter()
                    .map(move |name| ModelInfo {
                        id: name.clone(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: provider_name.clone(),
                        context_window: 4096,
                        max_output_tokens: 4096,
                        capabilities: vec!["chat".to_string()],
                        tier: "standard".to_string(),
                    })
                    .collect(),
                priority: 0,
                rate_limit_rpm: None,
            }
        })
        .collect();
    Ok(Json(ProvidersResponse { providers }))
}

/// Add provider
async fn add_provider(
    State(state): State<GatewayState>,
    Json(request): Json<AddProviderRequest>,
) -> Result<Json<ProviderResponse>, GatewayError> {
    let provider_type: ProviderType = request
        .provider_type
        .parse()
        .map_err(|e: String| GatewayError::Internal(anyhow::anyhow!(e)))?;
    let capabilities: Vec<ModelCapability> = request
        .capabilities
        .iter()
        .map(|c| {
            c.parse()
                .map_err(|e: String| GatewayError::Internal(anyhow::anyhow!(e)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tier: ModelTier = request.tier.parse().map_err(|e: String| {
        GatewayError::Internal(anyhow::anyhow!(e))
    })?;

    let provider_name = request.name.clone();
    let provider = state
        .router
        .add_provider(ProviderConfig {
            name: request.name,
            provider_type,
            api_key: request.api_key,
            base_url: request.base_url,
            models: vec![ModelConfig {
                name: provider_name.clone(),
                display_name: provider_name.clone(),
                context_window: request.context_window,
                max_output_tokens: request.max_output_tokens,
                cost_per_million_input: request.cost_per_million_input,
                cost_per_million_output: request.cost_per_million_output,
                capabilities,
                tier,
            }],
            enabled: request.enabled,
            priority: request.priority,
            rate_limit_rpm: request.rate_limit_rpm,
            timeout_secs: request.timeout_secs,
        })
        .await?;
    Ok(Json(ProviderResponse {
        provider: ProviderInfo {
            name: provider.name,
            provider_type: provider.provider_type,
            enabled: true,
            models: provider
                .models
                .into_iter()
                .map(move |name| ModelInfo {
                    id: name.clone(),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: provider_name.clone(),
                    context_window: 4096,
                    max_output_tokens: 4096,
                    capabilities: vec!["chat".to_string()],
                    tier: "standard".to_string(),
                })
                .collect(),
            priority: 0,
            rate_limit_rpm: None,
        },
    }))
}

/// Remove provider
async fn remove_provider(
    State(state): State<GatewayState>,
    Path(name): Path<String>,
) -> Result<Json<DeleteResponse>, GatewayError> {
    state.router.remove_provider(&name).await?;
    Ok(Json(DeleteResponse { success: true }))
}

/// Router status
async fn router_status(State(state): State<GatewayState>) -> Result<Json<RouterStatusResponse>, GatewayError> {
    let status = state.router.get_status().await?;
    Ok(Json(RouterStatusResponse {
        strategy: "round_robin".to_string(),
        active_provider: None,
        budget_used_today: 0.0,
        budget_limit_daily: 0.0,
        requests_today: 0,
        errors_today: 0,
        zero_config_enabled: status.zero_config,
    }))
}

/// Zero-config provider status
async fn zero_config_status(State(state): State<GatewayState>) -> Result<Json<ZeroConfigStatusResponse>, GatewayError> {
    let status = state.router.get_zero_config_status().await?;
    Ok(Json(ZeroConfigStatusResponse { providers: Vec::new() }))
}

/// WebSocket handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(_state): State<GatewayState>,
) -> Response {
    // Phase 0 stub: WebSocket agent streaming wired in Phase 1.
    ws.on_upgrade(|socket| async move {
        let mut socket = socket;
        let _ = socket.send(axum::extract::ws::Message::Text(
            serde_json::to_string(&serde_json::json!({"type":"info","message":"websocket streaming not implemented in Phase 0"})).unwrap(),
        )).await;
    })
}

async fn handle_websocket(_socket: axum::extract::ws::WebSocket, _state: GatewayState) {
    // Phase 0 stub: WebSocket handling wired in Phase 1.
}

// ── Request/Response Types ──

#[derive(Debug, Deserialize, Validate)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub capabilities: Vec<String>,
    pub tier: String,
}

#[derive(Debug, Deserialize)]
pub struct AgentRunRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<String>>,
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct AgentRunResponse {
    pub messages: Vec<ChatMessage>,
    pub tool_calls: Vec<ToolCall>,
    pub iterations: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
    pub result: Value,
    pub is_error: bool,
}

#[derive(Debug, Deserialize)]
pub struct AgentStreamQuery {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<String>>,
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct MemorySearchRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub threshold: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct MemorySearchResponse {
    pub results: Vec<MemorySearchResult>,
}

#[derive(Debug, Serialize)]
pub struct MemorySearchResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub metadata: Value,
    pub created_at: String,
}

#[derive(Debug, Deserialize, Validate)]
pub struct MemoryStoreRequest {
    pub content: String,
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct MemoryStoreResponse {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct ListConversationsQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SessionResetRequest {
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct MemoryContextQuery {
    pub query: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryStatsResponse {
    pub memory_count: usize,
    pub conversation_count: usize,
    pub message_count: usize,
}

#[derive(Debug, Serialize)]
pub struct ConversationsResponse {
    pub conversations: Vec<Conversation>,
}

#[derive(Debug, Serialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
    pub token_count: usize,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub success: bool,
}

#[derive(Debug, Serialize)]
pub struct McpServersResponse {
    pub servers: Vec<McpServerInfo>,
}

#[derive(Debug, Serialize)]
pub struct McpServerInfo {
    pub name: String,
    pub enabled: bool,
    pub transport: String,
    pub url: Option<String>,
    pub command: Option<String>,
    pub tools_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct AddMcpServerRequest {
    pub name: String,
    pub config: Value,
}

#[derive(Debug, Serialize)]
pub struct McpServerResponse {
    pub server: McpServerInfo,
}

#[derive(Debug, Serialize)]
pub struct McpToolsResponse {
    pub tools: Vec<McpToolInfo>,
}

#[derive(Debug, Serialize)]
pub struct McpToolInfo {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub read_only: bool,
}

#[derive(Debug, Deserialize)]
pub struct CallMcpToolRequest {
    pub server: String,
    pub tool: String,
    pub arguments: Value,
}

#[derive(Debug, Serialize)]
pub struct CallMcpToolResponse {
    pub result: Value,
}

#[derive(Debug, Deserialize)]
pub struct ScheduleRequest {
    pub message: String,
    pub natural_time: String,
}

#[derive(Debug, Deserialize)]
pub struct ScheduleRecurringRequest {
    pub message: String,
    pub natural_time: String,
}

#[derive(Debug, Serialize)]
pub struct ScheduleResponse {
    pub id: i64,
    pub message: String,
    pub trigger: String,
    pub backend: String,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListSchedulesQuery {
    pub active_only: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SchedulesResponse {
    pub schedules: Vec<ScheduleInfo>,
}

#[derive(Debug, Serialize)]
pub struct ScheduleInfo {
    pub id: i64,
    pub message: String,
    pub trigger: String,
    pub schedule_type: String,
    pub backend: String,
    pub created_at: String,
    pub active: bool,
}

#[derive(Debug, Serialize)]
pub struct ScheduleCancelResponse {
    pub id: i64,
    pub cancelled: bool,
}

#[derive(Debug, Serialize)]
pub struct ProvidersResponse {
    pub providers: Vec<ProviderInfo>,
}

#[derive(Debug, Serialize)]
pub struct ProviderInfo {
    pub name: String,
    pub provider_type: String,
    pub enabled: bool,
    pub models: Vec<ModelInfo>,
    pub priority: u8,
    pub rate_limit_rpm: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct AddProviderRequest {
    pub name: String,
    pub provider_type: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub cost_per_million_input: f64,
    pub cost_per_million_output: f64,
    pub capabilities: Vec<String>,
    pub tier: String,
    pub enabled: bool,
    pub priority: u8,
    pub rate_limit_rpm: Option<u32>,
    pub timeout_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct ProviderResponse {
    pub provider: ProviderInfo,
}

#[derive(Debug, Serialize)]
pub struct RouterStatusResponse {
    pub strategy: String,
    pub active_provider: Option<String>,
    pub budget_used_today: f64,
    pub budget_limit_daily: f64,
    pub requests_today: u64,
    pub errors_today: u64,
    pub zero_config_enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct ZeroConfigStatusResponse {
    pub providers: Vec<ZeroConfigProviderStatus>,
}

#[derive(Debug, Serialize)]
pub struct ZeroConfigProviderStatus {
    pub name: String,
    pub label: String,
    pub description: String,
    pub failures: u32,
    pub available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WsClientMessage {
    Ping,
    Chat {
        messages: Vec<ChatMessage>,
        model: String,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    },
    AgentRun {
        messages: Vec<ChatMessage>,
        tools: Option<Vec<String>>,
        max_iterations: Option<usize>,
    },
}

// ── Error Handling ──

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("Agent error: {0}")]
    Agent(#[from] crate::agent::AgentError),
    #[error("Memory error: {0}")]
    Memory(#[from] crate::memory::MemoryError),
    #[error("Router error: {0}")]
    Router(#[from] crate::router::RouterError),
    #[error("MCP error: {0}")]
    Mcp(#[from] crate::mcp::McpError),
    #[error("Scheduler error: {0}")]
    Scheduler(#[from] crate::scheduler::SchedulerError),
    #[error("Validation error: {0}")]
    Validation(#[from] validator::ValidationErrors),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Internal error: {0}")]
    Internal(anyhow::Error),
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            GatewayError::Validation(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            GatewayError::Agent(_) | GatewayError::Memory(_) | GatewayError::Router(_) | GatewayError::Mcp(_) | GatewayError::Scheduler(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            GatewayError::Serialization(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            GatewayError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string()),
        };

        let body = Json(serde_json::json!({
            "error": {
                "message": message,
                "type": std::any::type_name_of_val(&self).split("::").last().unwrap_or("Error"),
            }
        }));

        (status, body).into_response()
    }
}
