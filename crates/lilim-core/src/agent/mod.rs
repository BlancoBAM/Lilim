//! Agent Loop - autonomous agent execution with rig-core
//!
//! The agent targets exactly two endpoints, both OpenAI-compatible:
//!   1. Local mistral.rs (Phi-4-mini) when local-first/offline.
//!   2. The internal Axum OpenAI-compatible endpoint backed by litellm-rs for remote traffic.
//!
//! Tools are strongly-typed Rust structs implementing rig's `Tool` trait. Consequential
//! tools (shell writes, file mutations, external MCP write tools) are gated behind an
//! approval step that surfaces an approval request to the Tauri desktop UI.

use anyhow::Result;
use rig::{
    agent::AgentBuilder,
    client::CompletionClient,
    completion::{Chat, Message},
    providers::openai::{Client, ClientBuilder},
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use crate::{
    config::AgentConfig,
    memory::Memory,
    mcp::McpHost,
    router::Router,
};

/// Concrete rig agent type: OpenAI-compatible responses model (mistral.rs or litellm endpoint).
type RigAgent = rig::agent::Agent<
    rig::providers::openai::responses_api::ResponsesCompletionModel,
>;

/// Agent errors
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Completion error: {0}")]
    Completion(String),
    #[error("Tool execution error: {0}")]
    Tool(String),
    #[error("Approval required for tool: {0}")]
    ApprovalRequired(String),
    #[error("Memory error: {0}")]
    Memory(#[from] crate::memory::MemoryError),
}

/// Shared tool arguments/output (JSON) for flexible LLM tool calling.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ToolArgs(pub Value);

#[derive(Debug, Serialize, Clone)]
pub struct ToolOutput(pub Value);

/// Result returned when a consequential tool requires user approval before executing.
#[derive(Debug, Serialize, Clone)]
pub struct ApprovalRequest {
    pub tool: String,
    pub args: Value,
    pub reason: String,
}

/// Main agent
pub struct Agent {
    inner: RigAgent,
    config: AgentConfig,
    memory: Arc<Memory>,
    mcp: Arc<McpHost>,
    router: Arc<Router>,
}

impl Agent {
    /// Create a new agent backed by the rig OpenAI-compatible client.
    pub async fn new(
        router: Arc<Router>,
        memory: Arc<Memory>,
        mcp: Arc<McpHost>,
        config: &AgentConfig,
    ) -> Result<Self> {
        // Target the local mistral.rs endpoint by default; the gateway can rebuild
        // against the internal litellm endpoint when remote traffic is needed.
        let base_url = "http://127.0.0.1:11434/v1";
        let model = "phi-4-mini";
        let client = ClientBuilder::new("local")
            .base_url(base_url)
            .build()
            .map_err(|e| AgentError::Completion(format!("client builder: {e}")))?;
        let rig_model = client.completion_model(model);

        let memory_tool = MemorySearchTool { memory: memory.clone() };
        let shell_tool = ShellExecTool { approval: config.tool_approval_required.contains(&"shell_exec".to_string()) };
        let file_write_tool = FileWriteTool { approval: config.tool_approval_required.contains(&"file_write".to_string()) };
        let file_read_tool = FileReadTool;
        let schedule_tool = ScheduleTool { scheduler: None };

        let inner = AgentBuilder::new(rig_model)
            .tool(memory_tool)
            .tool(shell_tool)
            .tool(file_write_tool)
            .tool(file_read_tool)
            .tool(schedule_tool)
            .build();

        Ok(Self {
            inner,
            config: config.clone(),
            memory,
            mcp,
            router,
        })
    }

    /// Run the agent loop with a user prompt and optional chat history.
    /// Returns the final response text, or an approval-request string when a
    /// consequential tool needs user confirmation.
    #[tracing::instrument(skip(self, chat_history))]
    pub async fn run(&self, prompt: &str, chat_history: Vec<Message>) -> Result<String, AgentError> {
        let response = self
            .inner
            .chat(prompt.to_string(), chat_history)
            .await
            .map_err(|e| AgentError::Completion(e.to_string()))?;
        Ok(response)
    }

    /// Single-turn chat completion (prompt string in, text out).
    #[tracing::instrument(skip(self))]
    pub async fn chat(&self, prompt: &str) -> Result<String, AgentError> {
        self.run(prompt, Vec::new()).await
    }

    /// Single-turn completion (prompt string in, text out).
    #[tracing::instrument(skip(self))]
    pub async fn complete(&self, prompt: &str) -> Result<String, AgentError> {
        self.run(prompt, Vec::new()).await
    }

    /// Streaming endpoints are wired in Phase 1.
    pub async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _model: &str,
        _max_tokens: Option<usize>,
        _temperature: Option<f32>,
        _tx: tokio::sync::broadcast::Sender<String>,
    ) -> Result<(), AgentError> {
        Err(AgentError::Completion("streaming not implemented in Phase 0".to_string()))
    }

    pub async fn stream_run(
        &self,
        _messages: Vec<Message>,
        _tools: Vec<rig::completion::ToolDefinition>,
        _max_iterations: Option<usize>,
        _tx: tokio::sync::broadcast::Sender<String>,
    ) -> Result<(), AgentError> {
        Err(AgentError::Completion("streaming not implemented in Phase 0".to_string()))
    }

    pub async fn stream_chat_ws(
        &self,
        _messages: Vec<Message>,
        _model: &str,
        _max_tokens: Option<usize>,
        _temperature: Option<f32>,
        _tx: tokio::sync::broadcast::Sender<String>,
    ) -> Result<(), AgentError> {
        Err(AgentError::Completion("streaming not implemented in Phase 0".to_string()))
    }

    pub async fn stream_run_ws(
        &self,
        _messages: Vec<Message>,
        _tools: Vec<rig::completion::ToolDefinition>,
        _max_iterations: Option<usize>,
        _tx: tokio::sync::broadcast::Sender<String>,
    ) -> Result<(), AgentError> {
        Err(AgentError::Completion("streaming not implemented in Phase 0".to_string()))
    }

    /// Submit an approval decision for a previously-requested tool call.
    pub async fn approve(&self, _tool: &str, _approved: bool) -> Result<(), AgentError> {
        // Resume the pending turn. Implementation deferred to Phase 4 wiring.
        Ok(())
    }
}

// ─── Tools ───

/// Semantic memory recall tool (read-only: executes automatically).
#[derive(Clone)]
struct MemorySearchTool {
    memory: Arc<Memory>,
}

#[derive(Deserialize)]
struct MemorySearchArgs {
    query: String,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct MemorySearchOutput {
    results: Vec<crate::memory::MemorySearchResult>,
}

impl Tool for MemorySearchTool {
    const NAME: &'static str = "memory_search";
    type Error = AgentError;
    type Args = MemorySearchArgs;
    type Output = MemorySearchOutput;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search long-term memory for relevant past conversations and facts.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "limit": {"type": "number", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let results = self.memory.search(&args.query, args.limit, Some(0.5)).await?;
        Ok(MemorySearchOutput { results })
    }
}

/// Shell execution tool (consequential: requires approval when writes/mutations).
#[derive(Clone)]
struct ShellExecTool {
    approval: bool,
}

#[derive(Deserialize)]
struct ShellExecArgs {
    command: String,
}

#[derive(Serialize)]
struct ShellExecOutput {
    output: String,
}

impl Tool for ShellExecTool {
    const NAME: &'static str = "shell_exec";
    type Error = AgentError;
    type Args = ShellExecArgs;
    type Output = ShellExecOutput;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Execute a shell command. Consequential commands require approval.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if self.approval {
            return Err(AgentError::ApprovalRequired(format!(
                "shell_exec: {}",
                args.command
            )));
        }
        debug!("Executing shell command (read-only passthrough): {}", args.command);
        Ok(ShellExecOutput { output: format!("executed: {}", args.command) })
    }
}

/// File write tool (consequential: requires approval).
#[derive(Clone)]
struct FileWriteTool {
    approval: bool,
}

#[derive(Deserialize)]
struct FileWriteArgs {
    path: String,
    content: String,
}

#[derive(Serialize)]
struct FileWriteOutput {
    written: bool,
}

impl Tool for FileWriteTool {
    const NAME: &'static str = "file_write";
    type Error = AgentError;
    type Args = FileWriteArgs;
    type Output = FileWriteOutput;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Write content to a file. Requires approval.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "File content to write"}
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if self.approval {
            return Err(AgentError::ApprovalRequired(format!("file_write: {}", args.path)));
        }
        debug!("Writing file (placeholder): {}", args.path);
        Ok(FileWriteOutput { written: true })
    }
}

/// File read tool (read-only: executes automatically).
struct FileReadTool;

#[derive(Deserialize)]
struct FileReadArgs {
    path: String,
}

#[derive(Serialize)]
struct FileReadOutput {
    content: String,
}

impl Tool for FileReadTool {
    const NAME: &'static str = "file_read";
    type Error = AgentError;
    type Args = FileReadArgs;
    type Output = FileReadOutput;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read the contents of a file.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        debug!("Reading file (placeholder): {}", args.path);
        Ok(FileReadOutput { content: String::new() })
    }
}

/// Scheduled tasks tool.
#[derive(Clone)]
struct ScheduleTool {
    scheduler: Option<Arc<()>>,
}

#[derive(Deserialize)]
struct ScheduleArgs {
    message: String,
    natural_time: String,
}

#[derive(Serialize)]
struct ScheduleOutput {
    id: String,
}

impl Tool for ScheduleTool {
    const NAME: &'static str = "schedule_task";
    type Error = AgentError;
    type Args = ScheduleArgs;
    type Output = ScheduleOutput;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Schedule a one-time reminder or recurring task.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder message"},
                    "natural_time": {"type": "string", "description": "When to trigger (natural language)"}
                },
                "required": ["message", "natural_time"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let id = uuid::Uuid::new_v4().to_string();
        debug!("Scheduled task: {} at {}", args.message, args.natural_time);
        Ok(ScheduleOutput { id })
    }
}
