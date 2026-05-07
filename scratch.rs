use lilim_inference::{InferenceEngine, InferenceConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting");
    let mut config = InferenceConfig::default();
    config.max_gen_tokens = 5;
    let engine = InferenceEngine::new(config).await;
    println!("Engine loaded. Generating...");
    let result = engine.generate("Hello, how are you?", 5).await?;
    println!("Result: {}", result);
    Ok(())
}
