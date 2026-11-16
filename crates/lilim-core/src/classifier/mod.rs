//! Complexity Classifier - Smart routing based on request complexity

use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::config::ComplexityThresholds;

/// Classifier errors
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Classification error: {0}")]
    Classification(String),
    #[error("Model not loaded")]
    ModelNotLoaded,
}

/// Complexity classifier using embeddings
pub struct ComplexityClassifier {
    embedder: Arc<Mutex<TextEmbedding>>,
    thresholds: ComplexityThresholds,
    category_embeddings: Vec<(String, Vec<f32>)>,
}

impl ComplexityClassifier {
    /// Create a new classifier
    pub async fn new(embedding_model: &str) -> Result<Self> {
        let embedder = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| ClassifierError::Embedding(e.to_string()))?;
        let embedder = Arc::new(Mutex::new(embedder));

        // Pre-compute category embeddings
        let categories = vec![
            "general conversation chat",
            "coding programming software development",
            "reasoning logic analysis problem solving",
            "creative writing storytelling",
            "analysis research summarization",
            "quick simple question answer",
        ];

        let mut embedder_lock = embedder.lock().await;
        let embeddings = embedder_lock
            .embed(
                categories.iter().map(|s| s.to_string()).collect::<Vec<String>>(),
                None,
            )
            .map_err(|e| ClassifierError::Embedding(e.to_string()))?;
        drop(embedder_lock);

        let category_embeddings: Vec<(String, Vec<f32>)> = categories
            .into_iter()
            .zip(embeddings)
            .map(|(cat, emb)| (cat.to_string(), emb))
            .collect();

        Ok(Self {
            embedder,
            thresholds: ComplexityThresholds::default(),
            category_embeddings,
        })
    }

    /// Classify request complexity and category
    #[tracing::instrument(skip(self))]
    pub async fn classify(&self, messages: &[ChatMessage]) -> Result<ClassificationResult, ClassifierError> {
        // Combine recent messages for classification
        let text = messages.iter()
            .rev()
            .take(5)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if text.trim().is_empty() {
            return Ok(ClassificationResult {
                category: "general".to_string(),
                complexity: Complexity::Simple,
                confidence: 0.5,
                estimated_tokens: 100,
            });
        }

        // Get embedding
        let mut embedder_lock = self.embedder.lock().await;
        let embeddings = embedder_lock
            .embed(vec![text.clone()], None)
            .map_err(|e| ClassifierError::Embedding(e.to_string()))?;
        let query_embedding = embeddings.into_iter().next().unwrap_or_default();
        drop(embedder_lock);

        // Find best matching category
        let mut best_category = "general";
        let mut best_score = -1.0;

        for (category, cat_embedding) in &self.category_embeddings {
            let score = cosine_similarity(&query_embedding, cat_embedding);
            if score > best_score {
                best_score = score;
                best_category = category;
            }
        }

        // Map to our category names
        let category = map_category(best_category);

        // Estimate complexity based on multiple signals
        let complexity = self.estimate_complexity(&text, &messages).await;

        // Estimate token count
        let estimated_tokens = self.estimate_tokens(&text);

        debug!("Classified: category={}, complexity={:?}, tokens={}", category, complexity, estimated_tokens);

        Ok(ClassificationResult {
            category,
            complexity,
            confidence: best_score.max(0.0),
            estimated_tokens,
        })
    }

    /// Estimate complexity from text signals
    async fn estimate_complexity(&self, text: &str, messages: &[ChatMessage]) -> Complexity {
        let mut score = 0;

        // Length signal
        let word_count = text.split_whitespace().count();
        if word_count > 500 { score += 3; }
        else if word_count > 200 { score += 2; }
        else if word_count > 50 { score += 1; }

        // Technical terms
        let technical_terms = [
            "algorithm", "function", "class", "interface", "database", "api",
            "async", "await", "promise", "thread", "mutex", "lock",
            "optimize", "complexity", "architecture", "design pattern",
            "recursive", "dynamic programming", "graph", "tree",
        ];
        for term in technical_terms {
            if text.to_lowercase().contains(term) {
                score += 1;
            }
        }

        // Reasoning indicators
        let reasoning_terms = [
            "why", "how", "explain", "reason", "because", "therefore",
            "analyze", "compare", "evaluate", "prove", "derive",
            "step by step", "think through", "consider",
        ];
        for term in reasoning_terms {
            if text.to_lowercase().contains(term) {
                score += 1;
            }
        }

        // Code indicators
        if text.contains("```") || text.contains("def ") || text.contains("fn ") ||
           text.contains("function ") || text.contains("class ") || text.contains("import ") {
            score += 2;
        }

        // Multi-turn conversation
        if messages.len() > 5 {
            score += 1;
        }

        // Map score to complexity
        match score {
            0..=2 => Complexity::Simple,
            3..=5 => Complexity::Medium,
            6..=9 => Complexity::Complex,
            _ => Complexity::VeryComplex,
        }
    }

    /// Estimate token count
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimation: ~4 chars per token for English
        text.len() / 4 + 50
    }

    /// Classify with custom thresholds
    pub async fn classify_with_thresholds(
        &self,
        messages: &[ChatMessage],
        thresholds: &ComplexityThresholds,
    ) -> Result<ClassificationResult, ClassifierError> {
        let mut result = self.classify(messages).await?;

        // Override complexity based on token thresholds
        if result.estimated_tokens > thresholds.complex_max_tokens {
            result.complexity = Complexity::VeryComplex;
        } else if result.estimated_tokens > thresholds.medium_max_tokens {
            result.complexity = Complexity::Complex;
        } else if result.estimated_tokens > thresholds.simple_max_tokens {
            result.complexity = Complexity::Medium;
        }

        Ok(result)
    }
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub category: String,
    pub complexity: Complexity,
    pub confidence: f32,
    pub estimated_tokens: usize,
}

/// Complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

impl Complexity {
    pub fn max_tokens(&self, thresholds: &ComplexityThresholds) -> usize {
        match self {
            Complexity::Simple => thresholds.simple_max_tokens,
            Complexity::Medium => thresholds.medium_max_tokens,
            Complexity::Complex => thresholds.complex_max_tokens,
            Complexity::VeryComplex => thresholds.complex_max_tokens * 2,
        }
    }

    pub fn recommended_tier(&self) -> crate::config::ModelTier {
        match self {
            Complexity::Simple => crate::config::ModelTier::Free,
            Complexity::Medium => crate::config::ModelTier::Budget,
            Complexity::Complex => crate::config::ModelTier::Standard,
            Complexity::VeryComplex => crate::config::ModelTier::Premium,
        }
    }
}

/// Map internal category to routing category
fn map_category(internal: &str) -> String {
    if internal.contains("coding") || internal.contains("programming") {
        "coding".to_string()
    } else if internal.contains("reasoning") || internal.contains("logic") {
        "reasoning".to_string()
    } else if internal.contains("creative") || internal.contains("writing") {
        "creative".to_string()
    } else if internal.contains("analysis") || internal.contains("research") {
        "analysis".to_string()
    } else if internal.contains("quick") || internal.contains("simple") {
        "quick".to_string()
    } else {
        "general".to_string()
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Chat message for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}