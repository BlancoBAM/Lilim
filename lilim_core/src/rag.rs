use anyhow::Result;
use fastembed::{TextEmbedding, InitOptions};
use embedvec::{Distance, EmbedVec, BackendConfig, Quantization};
use tantivy::{schema::*, Index, query::QueryParser};
use tantivy::schema::Value;  // For .as_str() and .as_u64() methods
use serde_json::Value as JsonValue;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Category {
    Medical,
    SysAdmin,
    General,
}

#[derive(Debug, Clone)]
pub struct RagResult {
    pub text: String,
    pub category: String,
    pub authority: u8,
    pub score: f32,
    pub source: ResultSource,
}

#[derive(Debug, Clone)]
pub enum ResultSource {
    Keyword,
    Vector,
}

pub struct RagEngine {
    vector_db: EmbedVec,
    keyword_index: Index,
    embedder: TextEmbedding,
    schema: Schema,
}

impl RagEngine {
    pub fn new(vector_index_path: &Path, keyword_index_path: &Path) -> Result<Self> {
        // Load vector index
        let persistence = BackendConfig::new(vector_index_path.to_string_lossy());
        let vector_db = EmbedVec::new_internal(
            384,
            Distance::Cosine,
            32,
            200,
            Quantization::None,
            Some(persistence),
        )?;

        // Load keyword index
        let schema = build_schema();
        let keyword_index = Index::open_in_dir(keyword_index_path)?;

        // Initialize embedder
        let embedder = TextEmbedding::try_new(InitOptions::default())?;

        Ok(Self {
            vector_db,
            keyword_index,
            embedder,
            schema,
        })
    }

    /// Perform hybrid search with optional category filter
    pub async fn search(&mut self, query: &str, category_filter: Option<Category>, top_k: usize) -> Result<Vec<RagResult>> {
        let mut results = Vec::new();

        // Keyword search
        let reader = self.keyword_index.reader()?;
        let searcher = reader.searcher();
        
        let text_field = self.schema.get_field("text").unwrap();
        let category_field = self.schema.get_field("category").unwrap();
        let authority_field = self.schema.get_field("authority").unwrap();

        let query_parser = QueryParser::for_index(&self.keyword_index, vec![text_field]);
        let parsed_query = query_parser.parse_query(query)?;
        let keyword_hits = searcher.search(&parsed_query, &tantivy::collector::TopDocs::with_limit(top_k))?;

        for (score, doc_address) in keyword_hits {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;
            let text = doc.get_first(text_field).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let cat = doc.get_first(category_field).and_then(|v| v.as_str()).unwrap_or("General").to_string();
            let auth = doc.get_first(authority_field).and_then(|v| v.as_u64()).unwrap_or(1) as u8;

            // Apply category filter
            if let Some(ref filter) = category_filter {
                if !category_matches(&cat, filter) {
                    continue;
                }
            }

            results.push(RagResult {
                text,
                category: cat,
                authority: auth,
                score,
                source: ResultSource::Keyword,
            });
        }

        // Vector search
        let q_vec = self.embedder.embed(vec![query], None)?[0].clone();
        let vector_hits = self.vector_db.search(&q_vec, top_k, 100, None).await?;

        for hit in vector_hits {
            let text = hit.payload.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let cat = hit.payload.get("category").and_then(|v| v.as_str()).unwrap_or("General").to_string();
            let auth = hit.payload.get("authority").and_then(|v| v.as_u64()).unwrap_or(1) as u8;

            // Apply category filter
            if let Some(ref filter) = category_filter {
                if !category_matches(&cat, filter) {
                    continue;
                }
            }

            results.push(RagResult {
                text,
                category: cat,
                authority: auth,
                score: hit.score,
                source: ResultSource::Vector,
            });
        }

        // Fuse and rank by authority-weighted score
        Ok(fuse_and_rank(results))
    }

    /// Auto-detect domain from query
    pub fn detect_domain(&self, query: &str) -> Category {
        let query_lower = query.to_lowercase();
        
        let medical_keywords = ["anatomy", "medical", "diagnosis", "patient", "heart", "knee", "blood", "exam"];
        let sysadmin_keywords = ["sudo", "apt", "systemctl", "linux", "error", "wifi", "disk", "command"];

        let medical_count = medical_keywords.iter().filter(|k| query_lower.contains(*k)).count();
        let sysadmin_count = sysadmin_keywords.iter().filter(|k| query_lower.contains(*k)).count();

        if medical_count > sysadmin_count {
            Category::Medical
        } else if sysadmin_count > 0 {
            Category::SysAdmin
        } else {
            Category::General
        }
    }
}

fn category_matches(cat_str: &str, filter: &Category) -> bool {
    match filter {
        Category::Medical => cat_str == "Medical",
        Category::SysAdmin => cat_str == "SysAdmin",
        Category::General => true,
    }
}

fn fuse_and_rank(mut results: Vec<RagResult>) -> Vec<RagResult> {
    // Boost keyword matches and authority
    for result in &mut results {
        let authority_boost = result.authority as f32 / 5.0;
        let source_boost = match result.source {
            ResultSource::Keyword => 1.5,  // Prefer exact matches
            ResultSource::Vector => 1.0,
        };
        result.score *= authority_boost * source_boost;
    }

    // Sort by final score
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(5);  // Top 5 results
    results
}

fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("text", TEXT | STORED);
    schema_builder.add_text_field("category", STRING | STORED);
    schema_builder.add_u64_field("authority", STORED);
    schema_builder.add_u64_field("id", STORED);
    schema_builder.build()
}
