use anyhow::Result;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

pub struct MemoryEngine {
    conn: Connection,
}

impl MemoryEngine {
    /// Create new memory engine with SQLite database
    pub fn new(db_path: &str) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = Path::new(db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;
        
        // Initialize schema
        Self::init_schema(&conn)?;
        
        Ok(Self { conn })
    }

    /// Initialize database schema
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conv_messages ON messages(conversation_id, timestamp);
            "
        )?;
        Ok(())
    }

    /// Get or create conversation by session ID
    pub fn get_or_create_session(&self, session_id: &str) -> Result<i64> {
        // Try to get existing
        let mut stmt = self.conn.prepare(
            "SELECT id FROM conversations WHERE session_id = ?"
        )?;
        
        let result: Result<i64, _> = stmt.query_row(params![session_id], |row| row.get(0));
        
        match result {
            Ok(id) => Ok(id),
            Err(_) => {
                // Create new
                self.conn.execute(
                    "INSERT INTO conversations (session_id) VALUES (?)",
                    params![session_id],
                )?;
                Ok(self.conn.last_insert_rowid())
            }
        }
    }

    /// Save a message to conversation
    pub fn save_message(&self, conv_id: i64, role: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            params![conv_id, role, content],
        )?;
        Ok(())
    }

    /// Get conversation history (last N messages)
    pub fn get_history(&self, session_id: &str, limit: usize) -> Result<Vec<Message>> {
        let conv_id = self.get_or_create_session(session_id)?;
        
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content, timestamp 
             FROM messages 
             WHERE conversation_id = ? 
             ORDER BY timestamp DESC 
             LIMIT ?"
        )?;

        let messages = stmt.query_map(params![conv_id, limit], |row| {
            Ok(Message {
                id: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
                timestamp: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        // Reverse to get chronological order
        Ok(messages.into_iter().rev().collect())
    }

    /// Clean up old conversations
    pub fn cleanup_old_sessions(&self, days: i64) -> Result<usize> {
        let affected = self.conn.execute(
            "DELETE FROM conversations 
             WHERE created_at < datetime('now', ? || ' days')",
            params![format!("-{}", days)],
        )?;
        Ok(affected)
    }

    /// Get total message count for a session
    pub fn get_message_count(&self, session_id: &str) -> Result<usize> {
        let conv_id = self.get_or_create_session(session_id)?;
        
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            params![conv_id],
            |row| row.get(0),
        )?;
        
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let memory = MemoryEngine::new(":memory:").unwrap();
        let session_id = "test-session";
        
        let conv_id = memory.get_or_create_session(session_id).unwrap();
        assert!(conv_id > 0);
        
        // Save messages
        memory.save_message(conv_id, "user", "Hello").unwrap();
        memory.save_message(conv_id, "assistant", "Hi there!").unwrap();
        
        // Retrieve history
        let history = memory.get_history(session_id, 10).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].role, "assistant");
    }
}
