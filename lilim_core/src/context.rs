use crate::rag::{RagResult, Category};
use anyhow::Result;

pub struct ContextBuilder {
    system_prompt: String,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Self {
            system_prompt: Self::build_system_prompt(),
        }
    }

    fn build_system_prompt() -> String {
        r#"You are Lilim, a sarcastic but patient and reliable assistant embedded in Lilith Linux.

Your personality:
- 5% demonic, 5% infernal, 5% dark
- 25% caring, 25% wisely experienced, 25% askhole (dry, blunt, never hostile)
- Sarcasm is flavor, never friction

Communication Rules:
- Concise and clear by default
- Use "explain like I'm 10" for concepts
- Switch to verbose step-by-step for actions
- Never assume prior knowledge

Expertise:
- Ubuntu/Linux troubleshooting and optimization
- Medical Assistant coursework (A&P, terminology, clinical procedures)
- General first-year college subjects

Never provide unverified information. If uncertain, say so explicitly."#.to_string()
    }

    pub fn build_prompt_with_context(&self, user_query: &str, rag_results: &[RagResult], category: &Category) -> String {
        let mut prompt = self.system_prompt.clone();
        
        // Add domain context
        let domain_context = match category {
            Category::Medical => "\n\nCurrent Domain: Medical/Academic",
            Category::SysAdmin => "\n\nCurrent Domain: System Administration",
            Category::General => "\n\nCurrent Domain: General",
        };
        prompt.push_str(domain_context);

        // Add RAG context
        if !rag_results.is_empty() {
            prompt.push_str("\n\nRelevant Context from Knowledge Base:\n");
            for (i, result) in rag_results.iter().enumerate() {
                prompt.push_str(&format!(
                    "\n{}. [{}] (Authority: {})\n{}\n",
                    i + 1,
                    result.category,
                    result.authority,
                    result.text
                ));
            }
        }

        // Add user query
        prompt.push_str(&format!("\n\nUser Query: {}\n\nLilim's Response:", user_query));
        
        prompt
    }

    pub fn estimate_token_count(&self, text: &str) -> usize {
        // Rough estimate: ~4 chars per token
        text.len() / 4
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}
