//! Scheduler - Task scheduling via systemd-run or in-process
//!
//! Phase 0 stub: compiles and exposes the methods the gateway calls. Real
//! persistent scheduling storage is wired in Phase 4.

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::config::SchedulerConfig;

/// Scheduler errors
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("Cron parsing error: {0}")]
    CronParse(String),
    #[error("Systemd error: {0}")]
    Systemd(String),
    #[error("Time parsing error: {0}")]
    TimeParse(String),
    #[error("Notification error: {0}")]
    Notification(String),
    #[error("Schedule not found: {0}")]
    NotFound(String),
}

/// Schedule types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleType {
    Once,
    Recurring,
}

/// Schedule backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleBackend {
    Systemd,
    Threading,
}

/// A scheduled task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleInfo {
    pub id: i64,
    pub message: String,
    pub trigger: String,
    pub schedule_type: String,
    pub backend: String,
    pub created_at: String,
    pub active: bool,
}

/// Main Scheduler struct
pub struct Scheduler {
    config: SchedulerConfig,
}

impl Scheduler {
    pub async fn new(config: &SchedulerConfig) -> Result<Self, SchedulerError> {
        info!("Scheduler initialized (stub; persistent storage in Phase 4)");
        Ok(Self { config: config.clone() })
    }

    pub async fn start(&self) -> Result<(), SchedulerError> {
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<(), SchedulerError> {
        Ok(())
    }

    pub async fn schedule_once(
        &self,
        message: &str,
        natural_time: &str,
    ) -> Result<serde_json::Value, SchedulerError> {
        debug!("schedule_once (stub): {} at {}", message, natural_time);
        Ok(serde_json::json!({
            "id": 0,
            "message": message,
            "trigger": natural_time,
            "backend": "stub",
            "error": Option::<String>::None
        }))
    }

    pub async fn schedule_recurring(
        &self,
        message: &str,
        natural_time: &str,
    ) -> Result<serde_json::Value, SchedulerError> {
        debug!("schedule_recurring (stub): {} at {}", message, natural_time);
        Ok(serde_json::json!({
            "id": 0,
            "message": message,
            "trigger": natural_time,
            "backend": "stub",
            "error": Option::<String>::None
        }))
    }

    pub async fn list(&self, _active_only: bool) -> Result<Vec<ScheduleInfo>, SchedulerError> {
        Ok(Vec::new())
    }

    pub async fn cancel(&self, _id: i64) -> Result<serde_json::Value, SchedulerError> {
        Ok(serde_json::json!({ "id": 0, "cancelled": true }))
    }
}
