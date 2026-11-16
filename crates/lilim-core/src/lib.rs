//! Lilim Core - Pure Rust AI Agent Backend
//!
//! This crate replaces the entire Python backend (FastAPI, memory, routing, MCP, scheduling)
//! with a unified, high-performance Rust implementation.

pub mod agent;
pub mod classifier;
pub mod compression;
pub mod config;
pub mod gateway;
pub mod memory;
pub mod mcp;
pub mod router;
pub mod scheduler;

use anyhow::Result;
use std::sync::Arc;

/// Main LilimCore struct that holds all subsystem handles
pub struct LilimCore {
    pub gateway: Arc<gateway::Gateway>,
    pub agent: Arc<agent::Agent>,
    pub memory: Arc<memory::Memory>,
    pub router: Arc<router::Router>,
    pub mcp: Arc<mcp::McpHost>,
    pub scheduler: Arc<scheduler::Scheduler>,
}

impl LilimCore {
    /// Initialize all subsystems from configuration
    pub async fn new(config: config::Config) -> Result<Self> {
        let memory = Arc::new(memory::Memory::new(&config.memory).await?);
        let router = Arc::new(router::Router::new(&config.router).await?);
        let mcp = Arc::new(mcp::McpHost::new(&config.mcp).await?);
        let scheduler = Arc::new(scheduler::Scheduler::new(&config.scheduler).await?);
        let agent = Arc::new(agent::Agent::new(
            router.clone(),
            memory.clone(),
            mcp.clone(),
            &config.agent,
        ).await?);
        let gateway = Arc::new(gateway::Gateway::new(
            agent.clone(),
            memory.clone(),
            router.clone(),
            mcp.clone(),
            scheduler.clone(),
            &config,
        ).await?);

        Ok(Self {
            gateway,
            agent,
            memory,
            router,
            mcp,
            scheduler,
        })
    }

    /// Start all background tasks
    pub async fn start(&self) -> Result<()> {
        self.scheduler.start().await?;
        self.mcp.start().await?;
        Ok(())
    }

    /// Shutdown all subsystems gracefully
    pub async fn shutdown(&self) -> Result<()> {
        self.scheduler.shutdown().await?;
        self.mcp.shutdown().await?;
        self.memory.shutdown().await?;
        Ok(())
    }
}