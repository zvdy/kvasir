use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use kvasir::gpu::Gpu;
use kvasir::model::{Llama, LlamaConfig, Logits, Weights};
use kvasir::sampling::{Sampler, SamplingParams};
use serde::Deserialize;
use tokenizers::Tokenizer;

type BoxResult<T> = std::result::Result<T, Box<dyn Error>>;

const BOS_FALLBACK: &[&str] = &["<|begin_of_text|>", "<s>"];
const EOS_FALLBACK: &[&str] = &["<|end_of_text|>", "<|eot_id|>", "</s>"];

#[derive(Parser)]
#[command(
    name = "kvasir",
    version,
    about = "Llama 3.2 1B inference with hand-written CUDA kernels"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate text from a prompt.
    Generate(GenerateArgs),
}

#[derive(Args)]
struct GenerateArgs {
    /// Model directory: config.json, tokenizer.json, *.safetensors (generation_config.json optional).
    #[arg(long)]
    model: PathBuf,
    /// Prompt text; BOS is added once if the tokenizer does not add it.
    #[arg(long)]
    prompt: String,
    /// Maximum number of new tokens.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    /// RNG seed for sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Sampling temperature; 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Nucleus sampling mass in (0, 1].
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Do not stop at EOS tokens (fixed-length runs for benchmarking).
    #[arg(long)]
    ignore_eos: bool,
    /// Prompt tokens per prefill forward call.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u64).range(1..))]
    prefill_chunk: u64,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Print the final stats as one JSON line on stderr.
    #[arg(long)]
    json: bool,
}

#[derive(Deserialize, Default)]
struct GenerationConfig {
    bos_token_id: Option<u32>,
    eos_token_id: Option<OneOrMany>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Clone, Copy)]
enum Stop {
    Eos,
    MaxTokens,
    Context,
}

impl Stop {
    fn as_str(self) -> &'static str {
        match self {
            Stop::Eos => "eos",
            Stop::MaxTokens => "max_tokens",
            Stop::Context => "context",
        }
    }
}

struct Stats {
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill: Duration,
    decode_steps: usize,
    decode: Duration,
    stop: Stop,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Generate(a) => generate(&a),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn require(path: &Path, what: &str) -> BoxResult<()> {
    if !path.exists() {
        return Err(format!("{what} not found: {}", path.display()).into());
    }
    Ok(())
}

fn read_generation_config(dir: &Path) -> BoxResult<GenerationConfig> {
    let p = dir.join("generation_config.json");
    if !p.exists() {
        return Ok(GenerationConfig::default());
    }
    let text = std::fs::read_to_string(&p)?;
    Ok(serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?)
}

fn special_ids(cfg: &LlamaConfig, gen: GenerationConfig, tok: &Tokenizer) -> (Option<u32>, Vec<u32>) {
    let first_known = |names: &[&str]| names.iter().find_map(|n| tok.token_to_id(n));
    let bos = cfg
        .bos_token_id
        .or(gen.bos_token_id)
        .or_else(|| first_known(BOS_FALLBACK));
    let mut eos = cfg.eos_token_ids.clone();
    match gen.eos_token_id {
        Some(OneOrMany::One(x)) => eos.push(x),
        Some(OneOrMany::Many(v)) => eos.extend(v),
        None => {}
    }
    if eos.is_empty() {
        eos.extend(EOS_FALLBACK.iter().filter_map(|n| tok.token_to_id(n)));
    }
    eos.sort_unstable();
    eos.dedup();
    (bos, eos)
}

fn generate(a: &GenerateArgs) -> BoxResult<()> {
    let dir = &a.model;
    if !dir.is_dir() {
        return Err(format!("model directory not found: {}", dir.display()).into());
    }
    require(&dir.join("config.json"), "config.json")?;
    require(&dir.join("tokenizer.json"), "tokenizer.json")?;
    if a.max_tokens == 0 {
        return Err("--max-tokens must be at least 1".into());
    }
    let params = SamplingParams::new(a.temperature, a.top_p)?;

    let cfg = LlamaConfig::from_dir(dir)?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
    let (bos, eos) = special_ids(&cfg, read_generation_config(dir)?, &tok);
    let eos = if a.ignore_eos { Vec::new() } else { eos };

    let enc = tok
        .encode(a.prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let mut prompt: Vec<u32> = enc.get_ids().to_vec();
    if let Some(b) = bos {
        if prompt.first() != Some(&b) {
            prompt.insert(0, b);
        }
    }
    if prompt.is_empty() {
        return Err("prompt tokenized to zero tokens and the model has no BOS token".into());
    }
    if let Some(&bad) = prompt.iter().find(|&&t| t as usize >= cfg.vocab_size) {
        return Err(format!("token id {bad} out of range for vocab {}", cfg.vocab_size).into());
    }

    let limit = Llama::max_seq_limit(cfg.head_dim).min(cfg.max_position_embeddings);
    if prompt.len() >= limit {
        return Err(format!("prompt is {} tokens; the context limit is {limit}", prompt.len()).into());
    }
    let max_seq = (prompt.len().saturating_add(a.max_tokens - 1)).clamp(prompt.len(), limit);
    let chunk = usize::try_from(a.prefill_chunk)?.min(prompt.len());

    let gpu = Gpu::new(a.device)?;
    let weights = Weights::load(&gpu, dir, &cfg)?;
    let vocab = cfg.vocab_size;
    let mut model = Llama::new(&gpu, cfg, weights, max_seq, chunk, 1)?;
    let mut sampler = Sampler::new(params, vocab, a.seed)?;

    // Untimed warm-up of the prefill and decode shapes (cuBLAS lazy init), then start from an empty cache.
    model.forward(&gpu, &prompt[..chunk], Logits::Last)?;
    if model.len() < max_seq {
        model.forward(&gpu, &prompt[..1], Logits::Last)?;
    }
    model.reset();

    let t0 = Instant::now();
    let (head, last) = prompt.split_at(prompt.len() - (prompt.len() - 1) % chunk - 1);
    for part in head.chunks(chunk) {
        model.forward(&gpu, part, Logits::Last)?;
    }
    let logits = model.forward(&gpu, last, Logits::Last)?;
    gpu.synchronize()?;
    let prefill = t0.elapsed();
    let mut next = sampler.sample(logits)?;

    let mut out = std::io::stdout().lock();
    let mut generated: Vec<u32> = Vec::with_capacity(a.max_tokens);
    let mut printed = 0;
    let mut decode = Duration::ZERO;
    let mut decode_steps = 0;
    let stop = loop {
        generated.push(next);
        if eos.contains(&next) {
            break Stop::Eos;
        }
        let text = tok.decode(&generated, true).map_err(|e| format!("detokenize: {e}"))?;
        if text.len() > printed && !text.ends_with('\u{FFFD}') {
            if let Some(new) = text.get(printed..) {
                out.write_all(new.as_bytes())?;
                out.flush()?;
            }
            printed = text.len();
        }
        if generated.len() >= a.max_tokens {
            break Stop::MaxTokens;
        }
        if model.len() >= max_seq {
            break Stop::Context;
        }
        let t = Instant::now();
        next = sampler.sample(model.forward(&gpu, &[next], Logits::Last)?)?;
        gpu.synchronize()?;
        decode += t.elapsed();
        decode_steps += 1;
    };
    writeln!(out)?;
    out.flush()?;

    report(
        &Stats {
            prompt_tokens: prompt.len(),
            generated_tokens: generated.len(),
            prefill,
            decode_steps,
            decode,
            stop,
        },
        a.json,
    );
    Ok(())
}

fn per_sec(n: usize, d: Duration) -> f64 {
    let s = d.as_secs_f64();
    if s > 0.0 {
        n as f64 / s
    } else {
        0.0
    }
}

fn report(s: &Stats, json: bool) {
    let prefill_tps = per_sec(s.prompt_tokens, s.prefill);
    let decode_tps = per_sec(s.decode_steps, s.decode);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    if json {
        let v = serde_json::json!({
            "prompt_tokens": s.prompt_tokens,
            "generated_tokens": s.generated_tokens,
            "prefill_ms": ms(s.prefill),
            "prefill_tok_s": prefill_tps,
            "decode_steps": s.decode_steps,
            "decode_ms": ms(s.decode),
            "decode_tok_s": decode_tps,
            "stop": s.stop.as_str(),
        });
        eprintln!("{v}");
    } else {
        eprintln!(
            "prompt {} tok | generated {} tok (stop: {})\nprefill {:.2} ms, {:.1} tok/s | decode {} steps {:.2} ms, {:.1} tok/s",
            s.prompt_tokens,
            s.generated_tokens,
            s.stop.as_str(),
            ms(s.prefill),
            prefill_tps,
            s.decode_steps,
            ms(s.decode),
            decode_tps
        );
    }
}
