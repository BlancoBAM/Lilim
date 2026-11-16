//! Hybrid Memory System - SQLite + FastEmbed + pure-Rust cosine search
//!
//! Persistent hybrid memory stored entirely on disk under ~/.local/share/lilim/.
//! Relational data (conversations, messages, schedules) and vector embeddings live
//! in SQLite; semantic similarity is computed in Rust (cosine similarity), so there
//! is no external vector database and no C++ dependency.
//!
//! rusqlite's `Connection` is `!Send`/`!Sync`, so every SQLite operation runs inside
//! `tokio::task::spawn_blocking` on a fresh connection opened from the on-disk path.

use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::config::MemoryConfig;

/// Memory errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A stored memory with its embedding vector
struct MemoryRecord {
    id: String,
    content: String,
    metadata: Value,
    created_at: String,
    embedding: Vec<f32>,
}

/// Main Memory struct - SQLite + embedded vector search
pub struct Memory {
    config: MemoryConfig,
    sqlite_path: PathBuf,
    embedder: Arc<Mutex<TextEmbedding>>,
}

impl Memory {
    /// Create a new memory system
    pub async fn new(config: &MemoryConfig) -> Result<Self> {
        let sqlite_path = &config.sqlite_path;
        if let Some(parent) = sqlite_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let path = sqlite_path.clone();
        // Run schema creation on a fresh blocking connection
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            conn.execute_batch(SQLITE_SCHEMA)
        }).await.map_err(|e| MemoryError::Io(e.into()))??;

        let embedder = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;
        let embedder = Arc::new(Mutex::new(embedder));

        // Ensure vault directory exists
        std::fs::create_dir_all(&config.vault_path)?;

        info!("Memory system initialized at {}", sqlite_path.display());
        Ok(Self {
            config: config.clone(),
            sqlite_path: config.sqlite_path.clone(),
            embedder,
        })
    }

    /// Run a closure against a fresh SQLite connection on a blocking thread
    async fn sqlite<F, R>(&self, f: F) -> Result<R, MemoryError>
    where
        F: FnOnce(&mut Connection) -> Result<R, MemoryError> + Send + 'static,
        R: Send + 'static,
    {
        let path = self.sqlite_path.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = Connection::open(&path)?;
            f(&mut conn)
        })
        .await
        .map_err(|e| MemoryError::Io(e.into()))?
    }

    /// Generate an embedding for a piece of text
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let mut embedder = self.embedder.lock().await;
        let embeddings = embedder
            .embed(vec![text.to_string()], None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    /// Store a memory entry (SQLite row + embedding)
    #[tracing::instrument(skip(self))]
    pub async fn store(&self, content: &str, metadata: Option<Value>) -> Result<String, MemoryError> {
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let embedding = self.embed_text(content).await?;
        let embedding_json = serde_json::to_string(&embedding)?;
        let meta = metadata.clone().unwrap_or(Value::Null).to_string();
        let content = content.to_string();
        let id_for_insert = id.clone();

        self.sqlite(move |conn| {
            conn.execute(
                "INSERT INTO memories (id, content, metadata, embedding, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id_for_insert, content, meta, embedding_json, now, now],
            )?;
            Ok(())
        }).await?;
        debug!("Stored memory: {}", id);
        Ok(id)
    }

    /// Semantic search using cosine similarity over stored embeddings
    #[tracing::instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        limit: Option<usize>,
        threshold: Option<f32>,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        let query_embedding = self.embed_text(query).await?;
        let limit = limit.unwrap_or(10);
        let threshold = threshold.unwrap_or(0.0);

        let records = self.sqlite(move |conn| Self::load_records(conn)).await?;

        let mut scored: Vec<(f32, MemoryRecord)> = records
            .into_iter()
            .filter_map(|r| cosine_similarity(&query_embedding, &r.embedding).map(|s| (s, r)))
            .filter(|(s, _)| *s >= threshold)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .map(|(score, r)| MemorySearchResult {
                id: r.id,
                content: r.content,
                score,
                metadata: r.metadata,
                created_at: r.created_at,
            })
            .collect())
    }

    fn load_records(conn: &Connection) -> Result<Vec<MemoryRecord>, MemoryError> {
        let mut stmt = conn.prepare(
            "SELECT id, content, metadata, created_at, embedding FROM memories",
        )?;
        let rows = stmt.query_map([], |row| {
            let embedding: String = row.get(4)?;
            let embedding: Vec<f32> = serde_json::from_str(&embedding).unwrap_or_default();
            Ok(MemoryRecord {
                id: row.get(0)?,
                content: row.get(1)?,
                metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or(Value::Null),
                created_at: row.get(3)?,
                embedding,
            })
        })?;
        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }
        Ok(records)
    }

    /// Keyword search using SQLite FTS5
    pub async fn keyword_search(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        let limit = limit.unwrap_or(10);
        let query = query.to_string();

        self.sqlite(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, metadata, created_at FROM memories_fts WHERE memories_fts MATCH ?1 ORDER BY rank LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![query, limit], |row| {
                Ok(MemorySearchResult {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or(Value::Null),
                    created_at: row.get(3)?,
                    score: 1.0,
                })
            })?;
            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok(results)
        }).await
    }

    /// Hybrid search combining semantic and keyword results
    pub async fn hybrid_search(
        &self,
        query: &str,
        limit: Option<usize>,
        semantic_weight: f32,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        let limit = limit.unwrap_or(10);
        let semantic_results = self.search(query, Some(limit * 2), None).await?;
        let keyword_results = self.keyword_search(query, Some(limit * 2)).await?;

        let mut combined: std::collections::HashMap<String, MemorySearchResult> =
            std::collections::HashMap::new();

        for result in semantic_results.iter() {
            let score = result.score * semantic_weight;
            combined.insert(
                result.id.clone(),
                MemorySearchResult { score, ..result.clone() },
            );
        }
        for result in keyword_results.iter() {
            let n = keyword_results.len().max(1);
            let score = (1.0 - semantic_weight) * (1.0 - (0.0)); // rank-based handled below
            let _ = score;
            // Recompute with rank
        }
        // Rebuild with proper rank weighting
        let mut combined: std::collections::HashMap<String, MemorySearchResult> = std::collections::HashMap::new();
        for (i, result) in semantic_results.iter().enumerate() {
            combined.insert(result.id.clone(), MemorySearchResult {
                score: result.score * semantic_weight,
                ..result.clone()
            });
        }
        for (i, result) in keyword_results.iter().enumerate() {
            let score = (1.0 - semantic_weight) * (1.0 - (i as f32 / keyword_results.len().max(1) as f32));
            combined.entry(result.id.clone())
                .and_modify(|r| r.score += score)
                .or_insert(MemorySearchResult { score, ..result.clone() });
        }

        let mut results: Vec<_> = combined.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        Ok(results)
    }

    /// List conversations
    pub async fn list_conversations(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Conversation>, MemoryError> {
        let limit = limit.unwrap_or(50);
        let offset = offset.unwrap_or(0);
        self.sqlite(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, title, created_at, updated_at, token_count FROM conversations ORDER BY updated_at DESC LIMIT ?1 OFFSET ?2",
            )?;
            let rows = stmt.query_map(params![limit, offset], |row| {
                Ok(Conversation {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                    token_count: row.get(4)?,
                    messages: Vec::new(),
                })
            })?;
            let mut conversations = Vec::new();
            for row in rows {
                conversations.push(row?);
            }
            Ok(conversations)
        }).await
    }

    /// Get conversation by ID with messages
    pub async fn get_conversation(&self, id: &str) -> Result<Conversation, MemoryError> {
        let id = id.to_string();
        self.sqlite(move |conn| {
            let conv: Conversation = conn.query_row(
                "SELECT id, title, created_at, updated_at, token_count FROM conversations WHERE id = ?1",
                params![id.clone()],
                |row| Ok(Conversation {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                    token_count: row.get(4)?,
                    messages: Vec::new(),
                }),
            )?;

            let mut stmt = conn.prepare(
                "SELECT role, content, name, tool_calls, tool_call_id, created_at FROM messages WHERE conversation_id = ?1 ORDER BY created_at ASC",
            )?;
            let messages = stmt.query_map(params![id], |row| {
                Ok(ChatMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                    name: row.get(2)?,
                    tool_calls: row.get::<_, Option<String>>(3)?.map(|s| serde_json::from_str(&s).unwrap_or_default()),
                    tool_call_id: row.get(4)?,
                })
            })?.collect::<Result<Vec<_>, _>>()?;

            Ok(Conversation { messages, ..conv })
        }).await
    }

    /// Store a conversation
    pub async fn store_conversation(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        let conv = conversation.clone();
        self.sqlite(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![conv.id, conv.title, conv.created_at, conv.updated_at, conv.token_count],
            )?;
            tx.execute("DELETE FROM messages WHERE conversation_id = ?1", params![conv.id])?;
            for msg in &conv.messages {
                tx.execute(
                    "INSERT INTO messages (conversation_id, role, content, name, tool_calls, tool_call_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        conv.id,
                        msg.role,
                        msg.content,
                        msg.name,
                        msg.tool_calls.as_ref().map(|t| serde_json::to_string(t).unwrap()),
                        msg.tool_call_id,
                        chrono::Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            tx.commit()?;
            Ok(())
        }).await
    }

    /// Delete conversation
    pub async fn delete_conversation(&self, id: &str) -> Result<(), MemoryError> {
        let id = id.to_string();
        self.sqlite(move |conn| {
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM messages WHERE conversation_id = ?1", params![id])?;
            tx.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
            tx.commit()?;
            Ok(())
        }).await
    }

    /// Store a vault file (markdown) and index it for search
    pub async fn store_vault_file(&self, path: &str, content: &str) -> Result<(), MemoryError> {
        let full_path = self.config.vault_path.join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content_owned = content.to_string();
        std::fs::write(&full_path, &content_owned)?;

        let id = format!("vault::{}", path);
        let now = chrono::Utc::now().to_rfc3339();
        let mut metadata = serde_json::Map::new();
        metadata.insert("path".to_string(), Value::String(path.to_string()));
        metadata.insert("type".to_string(), Value::String("vault".to_string()));
        let embedding = self.embed_text(&content_owned).await?;
        let embedding_json = serde_json::to_string(&embedding)?;

        self.sqlite(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, metadata, embedding, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, content_owned, Value::Object(metadata).to_string(), embedding_json, now, now],
            )?;
            Ok(())
        }).await?;
        Ok(())
    }

    /// Search vault files
    pub async fn search_vault(&self, query: &str, limit: Option<usize>) -> Result<Vec<VaultSearchResult>, MemoryError> {
        let limit = limit.unwrap_or(10);
        let results = self.search(query, Some(limit), Some(0.6)).await?;
        Ok(results
            .into_iter()
            .filter(|r| r.metadata.get("type").and_then(|v| v.as_str()) == Some("vault"))
            .map(|r| VaultSearchResult {
                path: r.metadata.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                content: r.content,
                score: r.score,
            })
            .collect())
    }

    /// Schedule a one-time reminder (placeholder; scheduler owns scheduling)
    pub async fn schedule_once(&self, message: &str, natural_time: &str) -> Result<Value, MemoryError> {
        Ok(serde_json::json!({
            "id": Uuid::new_v4().to_string(),
            "message": message,
            "trigger": natural_time,
            "backend": "memory",
        }))
    }

    /// Schedule a recurring reminder (placeholder)
    pub async fn schedule_recurring(&self, message: &str, natural_time: &str) -> Result<Value, MemoryError> {
        Ok(serde_json::json!({
            "id": Uuid::new_v4().to_string(),
            "message": message,
            "trigger": natural_time,
            "backend": "memory",
        }))
    }

    /// List schedules (placeholder)
    pub async fn list_schedules(&self, _active_only: bool) -> Result<Vec<Value>, MemoryError> {
        Ok(vec![])
    }

    /// Cancel schedule (placeholder)
    pub async fn cancel_schedule(&self, _id: i64) -> Result<Value, MemoryError> {
        Ok(serde_json::json!({"cancelled": true}))
    }

    /// Load memory context for a query (returns concatenated relevant memories)
    pub async fn load_context(&self, query: &str) -> Result<String, MemoryError> {
        let results = self.search(query, Some(5), Some(0.3)).await?;
        if results.is_empty() {
            return Ok(String::new());
        }
        let mut context = String::from("Relevant memories:\n");
        for r in &results {
            context.push_str(&format!("- {} (score: {:.2})\n", r.content, r.score));
        }
        Ok(context)
    }

    /// Memory statistics
    pub async fn stats(&self) -> Result<MemoryStats, MemoryError> {
        let (memory_count, conversation_count, message_count) = self.sqlite(|conn| {
            let mc: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memories", [], |row| row.get(0)
            )?;
            let cc: i64 = conn.query_row(
                "SELECT COUNT(*) FROM conversations", [], |row| row.get(0)
            )?;
            let msgc: i64 = conn.query_row(
                "SELECT COUNT(*) FROM messages", [], |row| row.get(0)
            )?;
            Ok((mc, cc, msgc))
        }).await?;
        Ok(MemoryStats {
            memory_count: memory_count as usize,
            conversation_count: conversation_count as usize,
            message_count: message_count as usize,
        })
    }

    /// Shutdown and cleanup
    pub async fn shutdown(&self) -> Result<(), MemoryError> {
        info!("Shutting down memory system");
        Ok(())
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(dot / denom)
}

// ─── SQLite Schema ───

const SQLITE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    name TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
";

// ─── Types ───

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemorySearchResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub metadata: Value,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub token_count: usize,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VaultSearchResult {
    pub path: String,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryStats {
    pub memory_count: usize,
    pub conversation_count: usize,
    pub message_count: usize,
}
