//! MCP Host - Model Context Protocol client/server
//!
//! Phase 0 stub: compiles against the config types and exposes the methods the
//! gateway calls. Real rmcp transport lifecycle (stdio child processes +
//! streamable HTTP) is wired in Phase 4.
//!
//! Code-quality patterns adopted from paiml/paiml-mcp-agent-toolkit/docs/mcp/TOOLS.md:
//!   * `LIVE_MCP_TOOLS` is the single source of truth for advertised tools.
//!   * `tools_doc_states_the_live_tool_count` keeps the documented tool count in sync.
//!   * Every tool declares `read_only`; consequential tools are never silent about it.
//!   * Errors follow JSON-RPC convention with `data.suggestion` for recovery.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::config::McpConfig;

// ─── Live tool manifest (source of truth) ───

/// Declares one MCP tool offered by the Lilim host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolManifest {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    /// True when the tool performs no state mutation and may run without approval.
    pub read_only: bool,
}

/// Build the live tool manifest. Descriptions here must match what `list_tools`/`call_tool`
/// actually advertise. Update `DOCUMENTED_TOOL_COUNT` in the test below when
/// this list changes — the test asserts they stay in sync.
fn build_live_tools() -> Vec<McpToolManifest> {
    vec![
        McpToolManifest {
            name: "mcp_list_servers",
            description: "List MCP servers managed by the Lilim host (read-only).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            read_only: true,
        },
        McpToolManifest {
            name: "mcp_add_server",
            description: "Register a new MCP server (stdio child process or streamable HTTP). Consequential.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Unique server name" },
                    "config": { "description": "Server transport config (command/args or URL)" }
                },
                "required": ["name", "config"],
                "additionalProperties": false
            }),
            read_only: false,
        },
        McpToolManifest {
            name: "mcp_remove_server",
            description: "Remove a managed MCP server and stop its transport. Consequential.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Server name to remove" }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
            read_only: false,
        },
        McpToolManifest {
            name: "mcp_list_tools",
            description: "List all tools exposed by connected MCP servers (read-only).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            read_only: true,
        },
        McpToolManifest {
            name: "mcp_call_tool",
            description: "Invoke a tool on a connected MCP server. Consequential when the target tool mutates state.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": { "type": "string", "description": "Managed server name" },
                    "tool": { "type": "string", "description": "Tool name on that server" },
                    "arguments": { "description": "JSON arguments forwarded to the tool" }
                },
                "required": ["server", "tool", "arguments"],
                "additionalProperties": false
            }),
            read_only: false,
        },
    ]
}

/// Live tool manifest (source of truth). Lazily initialized because `Value`
/// cannot be constructed in a `const`.
static LIVE_MCP_TOOLS: once_cell::sync::Lazy<Vec<McpToolManifest>> =
    once_cell::sync::Lazy::new(build_live_tools);

/// Structured error payload carried by MCP errors for client-side recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpErrorData {
    pub code: i32,
    pub message: String,
    pub suggestion: Option<String>,
}

impl McpErrorData {
    fn invalid_params(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }
    fn not_found(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            code: -32601,
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: message.into(),
            suggestion: None,
        }
    }
}

/// MCP errors
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Transport error: {0}")]
    Transport(String),
    #[error("Tool call error: {0}")]
    ToolCall(String),
    #[error("Server not found: {0}")]
    ServerNotFound(String),
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        code: i32,
        message: String,
        suggestion: Option<String>,
    },
}

impl McpError {
    pub fn invalid_params(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        McpError::JsonRpc {
            code: -32602,
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }
    pub fn not_found(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        McpError::JsonRpc {
            code: -32601,
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        McpError::JsonRpc {
            code: -32603,
            message: message.into(),
            suggestion: None,
        }
    }
}

/// Describes an MCP server managed by the host
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    pub enabled: bool,
    pub transport: String,
    pub url: Option<String>,
    pub command: Option<String>,
    pub tools_count: usize,
}

/// Describes an MCP tool offered by a connected server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub read_only: bool,
}

/// MCP host - manages MCP server lifecycles and tool routing
pub struct McpHost {
    config: McpConfig,
    servers: Arc<RwLock<HashMap<String, McpServerInfo>>>,
}

impl McpHost {
    pub async fn new(config: &McpConfig) -> Result<Self, McpError> {
        info!("MCP host initialized (stub; rmcp wiring in Phase 4)");
        Ok(Self {
            config: config.clone(),
            servers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn start(&self) -> Result<(), McpError> {
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<(), McpError> {
        Ok(())
    }

    /// List managed MCP servers (read-only).
    pub async fn list_servers(&self) -> Result<Vec<McpServerInfo>, McpError> {
        Ok(self.servers.read().await.values().cloned().collect())
    }

    /// Register a new MCP server (consequential).
    pub async fn add_server(
        &self,
        name: String,
        _config: Value,
    ) -> Result<McpServerInfo, McpError> {
        if name.is_empty() {
            return Err(McpError::invalid_params(
                "server name must not be empty",
                "provide a non-empty unique server name",
            ));
        }
        let mut servers = self.servers.write().await;
        if servers.contains_key(&name) {
            return Err(McpError::invalid_params(
                format!("server '{name}' already exists"),
                "choose a different name or remove the existing server first",
            ));
        }
        let info = McpServerInfo {
            name: name.clone(),
            enabled: true,
            transport: "stdio".to_string(),
            url: None,
            command: None,
            tools_count: 0,
        };
        servers.insert(name.clone(), info.clone());
        Ok(info)
    }

    /// Remove a managed MCP server and stop its transport (consequential).
    pub async fn remove_server(&self, name: &str) -> Result<(), McpError> {
        let removed = self.servers.write().await.remove(name);
        if removed.is_none() {
            return Err(McpError::not_found(
                format!("server '{name}' not found"),
                "call mcp_list_servers to see registered server names",
            ));
        }
        Ok(())
    }

    /// List all tools exposed by connected MCP servers (read-only).
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        Ok(Vec::new())
    }

    /// Invoke a tool on a connected MCP server (consequential when the target mutates state).
    pub async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        arguments: Value,
    ) -> Result<Value, McpError> {
        let _ = arguments;
        if !self.servers.read().await.contains_key(server) {
            return Err(McpError::not_found(
                format!("server '{server}' not found"),
                "call mcp_list_servers to see registered server names",
            ));
        }
        // Stub: no real transport yet. Advertise that the call was not performed.
        Err(McpError::internal(format!(
            "mcp_call_tool not yet implemented for server '{server}', tool '{tool}' (rmcp wiring in Phase 4)"
        )))
    }

    /// Return the live tool manifest (read-only).
    pub fn live_tools(&self) -> &'static [McpToolManifest] {
        &LIVE_MCP_TOOLS
    }
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors paiml's `tools_doc_states_the_live_tool_count`:
    /// the documented tool count must stay in sync with `LIVE_MCP_TOOLS.len()`.
    #[test]
    fn tools_doc_states_the_live_tool_count() {
        const DOCUMENTED_TOOL_COUNT: usize = 5;
        assert_eq!(
            LIVE_MCP_TOOLS.len(),
            DOCUMENTED_TOOL_COUNT,
            "LIVE_MCP_TOOLS count drifted from DOCUMENTED_TOOL_COUNT; update both"
        );
    }

    #[test]
    fn live_tools_declare_read_only_flags() {
        for tool in LIVE_MCP_TOOLS.iter() {
            assert!(!tool.name.is_empty(), "tool name must not be empty");
            assert!(!tool.description.is_empty(), "tool description must not be empty");
            assert!(
                tool.input_schema.get("type").and_then(|t| t.as_str()) == Some("object"),
                "tool {} input_schema must be an object",
                tool.name
            );
        }
        let read_only: Vec<&str> = LIVE_MCP_TOOLS
            .iter()
            .filter(|t| t.read_only)
            .map(|t| t.name)
            .collect();
        assert!(
            read_only.contains(&"mcp_list_servers"),
            "mcp_list_servers must be read_only"
        );
        assert!(
            read_only.contains(&"mcp_list_tools"),
            "mcp_list_tools must be read_only"
        );
        assert!(
            !LIVE_MCP_TOOLS
                .iter()
                .any(|t| t.name == "mcp_add_server" && t.read_only),
            "mcp_add_server must be consequential"
        );
    }

    #[tokio::test]
    async fn add_server_rejects_empty_name() {
        let host = McpHost::new(&crate::config::McpConfig::default()).await.unwrap();
        let err = host.add_server(String::new(), serde_json::json!({})).await.unwrap_err();
        match err {
            McpError::JsonRpc { code, suggestion, .. } => {
                assert_eq!(code, -32602);
                assert!(suggestion.is_some());
            }
            other => panic!("expected JsonRpc invalid_params, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn remove_server_rejects_unknown() {
        let host = McpHost::new(&crate::config::McpConfig::default()).await.unwrap();
        let err = host.remove_server("no-such-server").await.unwrap_err();
        match err {
            McpError::JsonRpc { code, suggestion, .. } => {
                assert_eq!(code, -32601);
                assert!(suggestion.is_some());
            }
            other => panic!("expected JsonRpc not_found, got {other:?}"),
        }
    }
}
