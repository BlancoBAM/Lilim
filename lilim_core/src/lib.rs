// Re-export main modules
pub mod config;
pub mod context;
pub mod rag;
pub mod vlm;
pub mod api_client;
pub mod memory;

pub use rag::RagEngine;
pub use context::ContextBuilder;
pub use config::LilimConfig;
pub use vlm::{VLMEngine, SharedVLMEngine};
pub use memory::{MemoryEngine, Message};
