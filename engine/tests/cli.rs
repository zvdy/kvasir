use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use half::bf16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

const VOCAB: usize = 100;
const BOS: u32 = 0;
const EOS: u32 = 1;
const PROMPT: &str = "w5 w17 w42 w9 w63 w8 w77";

fn config_json() -> String {
    format!(
        r#"{{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
  "num_key_value_heads":2,"head_dim":16,"vocab_size":{VOCAB},"rms_norm_eps":1e-5,"rope_theta":10000.0,
  "max_position_embeddings":256,"tie_word_embeddings":true,"bos_token_id":{BOS},"eos_token_id":[{EOS}]}}"#
    )
}

fn write_weights(dir: &Path, seed: u64) {
    let (h, q, kv, inter, v) = (64usize, 64usize, 32usize, 128usize, VOCAB);
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![v, h]),
        ("model.norm.weight".into(), vec![h]),
    ];
    for i in 0..2 {
        for (n, s) in [
            ("input_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("post_attention_layernorm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![inter, h]),
            ("mlp.up_proj.weight", vec![inter, h]),
            ("mlp.down_proj.weight", vec![h, inter]),
        ] {
            shapes.push((format!("model.layers.{i}.{n}"), s));
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let raw: Vec<Vec<u8>> = shapes
        .iter()
        .map(|(n, s)| {
            let cnt = s.iter().product::<usize>();
            let (lo, hi) = if n.ends_with("norm.weight") {
                (0.5, 1.5)
            } else if n.contains("embed") {
                (-1.0, 1.0)
            } else {
                let a = (3.0 / s[1] as f32).sqrt();
                (-a, a)
            };
            (0..cnt)
                .flat_map(|_| bf16::from_f32(rng.random_range(lo..hi)).to_le_bytes())
                .collect()
        })
        .collect();
    let views: Vec<(String, TensorView)> = shapes
        .iter()
        .zip(&raw)
        .map(|((n, s), r)| (n.clone(), TensorView::new(Dtype::BF16, s.clone(), r).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
}

// WordLevel over "<s>", "</s>", "<unk>", "w3".."w99"; `with_bos` adds a "<s> $A" post-processor.
fn write_tokenizer(dir: &Path, with_bos: bool) {
    let mut vocab: HashMap<String, u32> = [("<s>", BOS), ("</s>", EOS), ("<unk>", 2)]
        .into_iter()
        .map(|(t, i)| (t.to_string(), i))
        .collect();
    vocab.extend((3..VOCAB as u32).map(|i| (format!("w{i}"), i)));
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".into())
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace {}));
    if with_bos {
        let tp = TemplateProcessing::builder()
            .try_single("<s> $A")
            .unwrap()
            .special_tokens(vec![("<s>", BOS)])
            .build()
            .unwrap();
        tok.with_post_processor(Some(tp));
    }
    tok.add_special_tokens([AddedToken::from("<s>", true), AddedToken::from("</s>", true)])
        .unwrap();
    tok.save(dir.join("tokenizer.json"), false).unwrap();
}

fn model_dir(name: &str, with_bos: bool) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_weights(&dir, 7);
    write_tokenizer(&dir, with_bos);
    dir
}

fn run(dir: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvasir"))
        .arg("generate")
        .arg("--model")
        .arg(dir)
        .args(["--prompt", PROMPT])
        .args(extra)
        .output()
        .unwrap()
}

fn ok_stats(out: &Output) -> serde_json::Value {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "kvasir failed: {stderr}");
    let line = stderr.lines().last().unwrap();
    serde_json::from_str(line).unwrap_or_else(|e| panic!("stats line {line:?}: {e}"))
}

fn words(out: &Output) -> Vec<String> {
    String::from_utf8(out.stdout.clone())
        .unwrap()
        .split_whitespace()
        .map(String::from)
        .collect()
}

#[test]
fn greedy_is_stable_and_reports_stats() {
    let dir = model_dir("greedy", true);
    let args = ["--max-tokens", "12", "--seed", "3", "--ignore-eos", "--json"];
    let a = run(&dir, &args);
    let b = run(&dir, &args);
    let s = ok_stats(&a);
    ok_stats(&b);
    assert_eq!(a.stdout, b.stdout);
    assert_eq!(s["prompt_tokens"], 8);
    assert_eq!(s["generated_tokens"], 12);
    assert_eq!(s["decode_steps"], 11);
    assert_eq!(s["stop"], "max_tokens");
    for k in ["prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s"] {
        assert!(s[k].as_f64().unwrap() > 0.0, "{k} = {}", s[k]);
    }
    let w = words(&a);
    assert_eq!(w.len(), 12);
    assert!(w.iter().all(|t| t.starts_with('w') || t == "<unk>"), "{w:?}");
}

#[test]
fn sampling_is_reproducible_per_seed() {
    let dir = model_dir("sampled", true);
    let go = |seed: &str, chunk: &str| {
        let out = run(
            &dir,
            &[
                "--max-tokens",
                "24",
                "--seed",
                seed,
                "--temperature",
                "1.5",
                "--top-p",
                "0.95",
                "--ignore-eos",
                "--json",
                "--prefill-chunk",
                chunk,
            ],
        );
        ok_stats(&out);
        out.stdout
    };
    let a = go("1", "512");
    assert_eq!(a, go("1", "512"));
    assert_eq!(a, go("1", "3"), "prefill chunking must not consume RNG draws");
    assert_ne!(a, go("2", "512"));
}

#[test]
fn bos_added_once_and_chunked_prefill() {
    let with = model_dir("bos_template", true);
    let without = model_dir("bos_none", false);
    let args = ["--max-tokens", "6", "--ignore-eos", "--json", "--prefill-chunk", "3"];
    let a = run(&with, &args);
    let b = run(&without, &args);
    assert_eq!(ok_stats(&a)["prompt_tokens"], 8);
    assert_eq!(ok_stats(&b)["prompt_tokens"], 8);
    assert_eq!(a.stdout, b.stdout);
}

#[test]
fn stops_at_eos_or_context() {
    let dir = model_dir("eos", true);
    let s = ok_stats(&run(&dir, &["--max-tokens", "200", "--json"]));
    let n = s["generated_tokens"].as_u64().unwrap();
    match s["stop"].as_str().unwrap() {
        "eos" => assert!(n <= 200),
        "max_tokens" => assert_eq!(n, 200),
        other => panic!("unexpected stop {other}"),
    }
    let s = ok_stats(&run(&dir, &["--max-tokens", "1000", "--ignore-eos", "--json"]));
    assert_eq!(s["stop"], "context");
    assert_eq!(
        s["prompt_tokens"].as_u64().unwrap() + s["decode_steps"].as_u64().unwrap(),
        256
    );
}

#[test]
fn human_stats_on_stderr() {
    let dir = model_dir("human", true);
    let out = run(&dir, &["--max-tokens", "4", "--ignore-eos"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("prompt 8 tok | generated 4 tok"), "{err}");
    assert!(err.contains("tok/s"), "{err}");
}

#[test]
fn missing_model_dir_fails_cleanly() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli_does_not_exist");
    let out = run(&dir, &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("model directory not found"));

    let empty = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli_empty");
    std::fs::create_dir_all(&empty).unwrap();
    let out = run(&empty, &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("config.json not found"));
}
