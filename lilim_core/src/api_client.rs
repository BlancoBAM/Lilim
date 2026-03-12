use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

use crate::config::ProviderConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub text: String,
    pub model: String,
    pub provider: String,
}

pub struct GenericAPIClient {
    client: Client,
}

impl GenericAPIClient {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
        }
    }

    /// Generic provider query - adapts to different API formats
    pub async fn query(
        &self,
        provider: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse> {
        let api_key = provider
            .api_key_env
            .as_ref()
            .and_then(|env| std::env::var(env).ok());

        // Build request based on provider type
        let body = self.build_request_body(provider, request)?;
        
        // Build HTTP client request
        let mut req_builder = self
            .client
            .post(&provider.api_url)
            .timeout(Duration::from_secs(provider.timeout_s))
            .json(&body);

        // Add authentication
        req_builder = match provider.auth_type.as_str() {
            "bearer" => {
                if let Some(key) = api_key {
                    req_builder.bearer_auth(key)
                } else {
                    return Err(anyhow!("API key not found for provider: {}", provider.name));
                }
            }
            "api-key" => {
                if let Some(key) = api_key {
                    req_builder.header("X-API-Key", key)
                } else {
                    return Err(anyhow!("API key not found for provider: {}", provider.name));
                }
            }
            "custom" => {
                // Add custom headers with environment variable substitution
                for (key, value) in &provider.custom_headers {
                    let value = self.substitute_env_vars(value);
                    req_builder = req_builder.header(key, value);
                }
                req_builder
            }
            "none" => req_builder,
            _ => return Err(anyhow!("Unknown auth type: {}", provider.auth_type)),
        };

        // Send request
        let response = req_builder.send().await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Provider {} returned error {}: {}",
                provider.name,
                status,
                error_text
            ));
        }

        let response_json: Value = response.json().await?;
        let text = self.extract_response_text(&response_json, provider)?;

        Ok(ProviderResponse {
            text,
            model: provider.model.clone(),
            provider: provider.name.clone(),
        })
    }

    /// Build request body adapted to provider format
    fn build_request_body(
        &self,
        provider: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<Value> {
        // Detect provider type from API URL
        let body = if provider.api_url.contains("openai.com") {
            // OpenAI format
            let mut messages = vec![];
            if let Some(sys) = &request.system_prompt {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": sys
                }));
            }
            messages.push(serde_json::json!({
                "role": "user",
                "content": request.prompt
            }));

            serde_json::json!({
                "model": provider.model,
                "messages": messages,
                "max_tokens": request.max_tokens.or(provider.max_tokens),
                "temperature": request.temperature.or(provider.temperature),
            })
        } else if provider.api_url.contains("anthropic.com") {
            // Anthropic format
            serde_json::json!({
                "model": provider.model,
                "messages": [{
                    "role": "user",
                    "content": request.prompt
                }],
                "system": request.system_prompt.clone().unwrap_or_default(),
                "max_tokens": request.max_tokens.or(provider.max_tokens).unwrap_or(4096),
                "temperature": request.temperature.or(provider.temperature),
            })
        } else if provider.api_url.contains("ollama") {
            // Ollama format
            serde_json::json!({
                "model": provider.model,
                "prompt": request.prompt,
                "system": request.system_prompt,
                "stream": provider.stream.unwrap_or(false),
            })
        } else {
            // Generic format (OpenAI-compatible)
            let mut messages = vec![];
            if let Some(sys) = &request.system_prompt {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": sys
                }));
            }
            messages.push(serde_json::json!({
                "role": "user",
                "content": request.prompt
            }));

            serde_json::json!({
                "model": provider.model,
                "messages": messages,
                "max_tokens": request.max_tokens.or(provider.max_tokens),
                "temperature": request.temperature.or(provider.temperature),
            })
        };

        Ok(body)
    }

    /// Extract response text from different API formats
    fn extract_response_text(&self, json: &Value, provider: &ProviderConfig) -> Result<String> {
        // Try different response formats
        
        // OpenAI format: choices[0].message.content
        if let Some(content) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
        {
            return Ok(content.to_string());
        }

        // Anthropic format: content[0].text
        if let Some(text) = json
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            return Ok(text.to_string());
        }

        // Ollama format: response
        if let Some(response) = json.get("response").and_then(|r| r.as_str()) {
            return Ok(response.to_string());
        }

        // Generic: text field
        if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
            return Ok(text.to_string());
        }

        Err(anyhow!(
            "Could not extract response text from provider {}. Response: {:?}",
            provider.name,
            json
        ))
    }

    /// Substitute environment variables in string (e.g., "${VAR_NAME}")
    fn substitute_env_vars(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Simple regex-free implementation
        while let Some(start) = result.find("${") {
            if let Some(end) = result[start..].find('}') {
                let var_name = &result[start + 2..start + end];
                let value = std::env::var(var_name).unwrap_or_default();
                result.replace_range(start..start + end + 1, &value);
            } else {
                break;
            }
        }
        
        result
    }
}
