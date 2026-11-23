# kvasir — design notes (Stage 1: Llama 3.2 1B, batch 1)

Target: one RTX 4070 (12 GB, sm_89, ~504 GB/s). Rust + hand-written CUDA kernels, cuBLAS only for GEMMs.

## 1. Module boundaries

```
engine/
  build.rs            nvcc -ptx -arch=sm_89 for kernels/*.cu -> $OUT_DIR/*.ptx (embedded with include_str!)
  kernels/            embedding.cu rmsnorm.cu rope.cu attention.cu silu_mul.cu elementwise.cu common.cuh
  src/error.rs        KvasirError (+ ensure_shape! macro). No unwrap/expect outside tests and main.
  src/gpu/
    device.rs         Gpu { ctx, stream, blas, kernels } — one context, ONE stream (stage 1)
    buffer.rs         DeviceTensor<T>: owned CudaSlice + logical shape, deliberately not Clone
    kernels.rs        one shape-checked launch wrapper per kernel, as methods on Gpu
    gemm.rs           linear_f16 / linear_f32: cublasGemmEx, FP16 in, FP32 accumulate
  src/model/
    config.rs         LlamaConfig from config.json (validated; rope_type must be "llama3" or absent)
    rope.rs           host cos/sin tables incl. llama3 long-context frequency scaling
    weights.rs        mmap safetensors -> GPU, BF16->FP16 on device, QKV and gate/up fused
    llama.rs          Llama { cfg, Weights, KvCache, Workspace }: forward(ids, Logits::Last | All)
    workspace.rs      Workspace: preallocated activations; WorkspaceDims = pure size math
  src/kv/             KvCache: contiguous K/V, per-layer views, length + bounds tracking
  src/sampling.rs     greedy / temperature / top-p on host (Sampler, reused scratch, seeded StdRng)
  src/bin/kvasir.rs   `kvasir generate`: tokenize, chunked prefill, decode loop, streamed text, tok/s on stderr
engine/tests/         kernels.rs (per-kernel vs CPU), weights.rs (synthetic checkpoint), workspace.rs, kv.rs (cache placement),
                      forward.rs (tiny model vs CPU f64 reference forward)
tests/ , ref/         logit-matching test and HF reference dumper        [not written yet]
```

Ownership: device memory is owned by exactly three long-lived structs — `Weights`, `KvCache`,
`Workspace`. Kernel wrappers take borrowed `CudaView`/`CudaViewMut`, never allocate. The generation
loop therefore performs no device allocation (the logits D2H copy goes into a reused host `Vec`).

## 2. Data layouts

Conventions: everything is row-major, contiguous, FP16 unless stated. `T` = number of tokens in the call.

| tensor | shape | notes |
|---|---|---|
| hidden / activations | `[T, 2048]` | one token per row |
| `W_qkv` (fused) | `[3072, 2048]` | rows = `q (32×64) \| k (8×64) \| v (8×64)` |
| `W_o` | `[2048, 2048]` | HF layout `[out, in]` |
| `W_gu` (fused) | `[16384, 2048]` | rows = `gate (8192) \| up (8192)` |
| `W_down` | `[2048, 8192]` | |
| `embed` (= LM head, tied) | `[128256, 2048]` | read as a gather by the embedding and as a matrix by the head |
| norm weights | `[2048]` | |
| QKV activation | `[T, 3072]` | same `q \| k \| v` order as the fused weight rows |
| attention out | `[T, 2048]` | head `h` at columns `[64h, 64h+64)` |
| gate/up activation | `[T, 16384]` | `silu_mul` reads it, writes `[T, 8192]` |
| logits | `[T_head, 128256]` FP32 | never rounded to FP16 |
| RoPE cos/sin | `[max_seq, 32]` FP32 | built on host, uploaded once by `KvCache::from_config`, which owns them |

**Workspace.** Allocated once for `max_tokens` (T upper bound) and `max_head_rows` (T_head upper bound, 1 for
generation): `ids [T] i32`, `hidden`/`normed`/`proj [T, hidden]` (`proj` = o_proj / down_proj output before the
residual add), `qkv [T, qkv_dim]`, `attn [T, q_dim]`, `gate_up [T, 2·inter]`, `act [T, inter]`, `logits [T_head, vocab]`
FP32, plus host buffers for the ids and the logits copy. A call with `t` tokens uses the leading `t` rows of each
buffer. Llama 3.2 1B: 70 KiB per token of FP16 activations + 501 KiB per head row.

**Linear layers.** Weights stay in HF `[out, in]` layout. `Y[T,out] = X[T,in] · Wᵀ` is issued to
cuBLAS (column-major) as `C[out×T] = op_T(A) · B` with `A = W` (`[in×out]` col-major, lda = in), `B = X`
(`[in×T]`, ldb = in), ldc = out. The result is already row-major `[T, out]`; nothing is transposed or copied.

**Head layout.** Within a token row, head `h` occupies a contiguous 64-element span. GQA: query head `h` uses
KV head `h / 4` (32 query heads, 8 KV heads).

**RoPE.** HF `rotate_half` (non-interleaved) convention: rotation pairs are `(j, j+32)` inside each head. This
matches the HF checkpoint's q/k row ordering (the converter permutes rows to this form), so weights are used
as-is. Position of token `i` in a call = `pos0 + i`. Frequencies follow HF's `llama3` rope scaling
(factor 32, low/high freq factors 1/4, original max 8192, θ = 500000); verified in `model::rope` unit tests for the
three wavelength regimes. Applied in place to the q and k slices of the QKV buffer, before the KV append.

**KV cache (stage 1, contiguous).** Two tensors (K, V), each `[layers=16, n_kv=8, max_seq, head_dim=64]` FP16.
Per layer/head the keys are contiguous over positions, so the attention kernel streams a head's keys linearly.
`max_seq` is chosen at startup (default 4096 → 2 × 16·8·4096·64·2 B = 128 MiB total). `kv_append` writes the
(already RoPE'd) K and raw V of the new tokens at `pos0..pos0+T`. This maps directly onto stage 3's paged blocks
(`[block, kv_head, block_size, head_dim]`): only the address computation in `attention`/`kv_append` changes.
`KvCache` (`src/kv/`) hands the kernels layer `l` as the contiguous `[n_kv, max_seq, head_dim]` slice at offset
`l·n_kv·max_seq·head_dim`, which is exactly the per-layer layout both wrappers check. Step protocol: during a forward
call `pos0 = cache.len()`; each layer calls `append` then `attention` (both use `pos0`), and `advance(T)` commits
once after the last layer. Every call checks `len + T ≤ max_seq` and returns `KvasirError::Shape` on overflow.

**RoPE table ownership.** The cos/sin tables live in `KvCache` (built by `from_config`, `[max_seq, head_dim/2]` each),
not in `Workspace`: their row count is the cache capacity `max_seq`, not `max_tokens`, and every position a
forward call can rotate is exactly a position the cache can hold, so one bound covers both. `KvCache::new` (generic
dimensions, no config) builds none and `rope()` then returns an error. 2 × 4096 × 32 × 4 B = 1 MiB for Llama 3.2 1B.

**Prefill vs decode.** Same layer code and same kernels; `T = prompt_len, pos0 = 0` vs `T = 1, pos0 = cur_len`.
Query `i` attends to keys `[0, pos0+i]`. Only the *last* row goes through the final norm and LM head during
generation (the logit-matching test asks for all rows).

**Forward pass (`model::Llama`).** `Llama` owns `Weights`, `KvCache` and `Workspace` and holds no other device
memory. `forward(gpu, ids, Logits::Last | Logits::All)` runs `ids` at positions `len..len+T` and returns a host
slice of FP32 logits (`[1, vocab]` or `[T, vocab]`, borrowed from the workspace's reused host buffer). Per layer:
`rmsnorm → W_qkv → rope_inplace → cache.append → cache.attention → W_o → hidden += proj → rmsnorm → W_gu →
silu_mul → W_down → hidden += proj`; then `cache.advance(T)`, the final norm on the last `T_head` rows only (written
to the leading rows of `normed`) and the LM head GEMM into `logits`. All views are leading-row slices
(`DeviceTensor::row_view`); nothing is allocated per call. Inputs (`1 ≤ T ≤ max_tokens`, `T_head ≤ max_head_rows`,
`len + T ≤ max_seq`, ids `< vocab`) are validated before any device work, so a rejected call leaves the sequence
unchanged; `len` only advances after the last layer. `Llama::new` rejects `max_tokens > max_seq` and a `max_seq`
above the attention kernel's shared-memory limit (`(max_seq + head_dim + 128)·4 B ≤ 48 KiB`, 12 096 for
head_dim 64), so the limit surfaces at construction instead of mid-generation. `reset`/`truncate` rewind the cache.
`Llama::max_seq_limit(head_dim)` exposes that bound.

**Generation (`kvasir generate`).** The prompt is encoded with the tokenizer's post-processor (Llama 3 adds BOS
128000); BOS is prepended only if the first id is not already BOS. EOS ids are the union of `config.json` and
`generation_config.json` `eos_token_id` (number or list; tokenizer names as fallback), so instruct checkpoints stop
on 128001/128008/128009. `max_seq = min(prompt + max_tokens − 1, max_position_embeddings, max_seq_limit)`,
`max_tokens` (workspace) = `min(prefill_chunk, prompt)`, `max_head_rows = 1`. The prompt is fed in chunks with
`Logits::Last`, then `forward(&[tok], Logits::Last) → Sampler::sample` until EOS, `--max-tokens` or a full cache.
One untimed warm-up (one prefill chunk and one decode step, then `reset`) absorbs cuBLAS lazy init. Text is
streamed by decoding the whole generated id list and printing the new suffix, held back while it ends in U+FFFD
(partial UTF-8 byte tokens). Timers: prefill = prompt forward calls; decode = each decode `forward + sample` with
`gpu.synchronize()` before the timer is read; decode tok/s = decode steps / decode time (the first token comes from
prefill). Stats go to stderr (`--json`: one JSON line).

## 3. Kernels (naive, correct, validated against CPU before use)

| kernel | grid × block | notes |
|---|---|---|
| `embedding_gather` | T × 256 | ids range-checked on host before upload |
| `rmsnorm` | rows × 256 | FP32 sum of squares; block reduce via warp shuffles |
| `rope_inplace` | T × 128 | one thread per (head, pair) |
| `kv_append` | T × 128 | |
| `attention` | (32, T) × 128 | block per (head, query). Scores in dynamic smem (FP32), 3 passes (max, exp+sum, P·V). Needs `(pos0+T+64+128)·4 B ≤ 48 KiB` → kv length ≲ 11.9k; wrapper errors beyond that |
| `silu_mul` | grid-stride | |
| `add_inplace` | grid-stride | residual add |
| `bf16_to_f16` | grid-stride | counts values outside FP16 range |

Unit-test results (`cargo test --test kernels`, 16 tests): all outputs are within one FP16 output rounding
(≈ 4.9e-4 relative) of an f32/f64 CPU reference; FP32-output GEMM is within 7e-7 absolute. Tolerances in the tests
are `1.5–2e-3 rel + 1e-3 abs`, i.e. ~3–4× the rounding floor, so a wrong index or missing term fails loudly.

## 4. Numerics: deviations from HF

1. **Weights/activations FP16, HF reference FP32.** Checkpoint is BF16; we convert to FP16 on the GPU (BF16 has
   8 exponent bits, FP16 only 5, so a load-time counter rejects any |w| > 65504 or non-finite value; mantissa goes
   8 → 11 bits, so in-range values convert exactly). Expected end-to-end logit error is dominated by FP16
   activations; the integration test calibrates its threshold against HF-FP16-vs-HF-FP32 (see §6).
2. **RMSNorm:** HF rounds `x·rsqrt(..)` to the activation dtype, then multiplies by the weight in that dtype.
   We keep the product in FP32 and round once. Strictly more accurate than HF-FP16, same as HF-FP32 in the limit.
3. **Fused projections:** `q|k|v` and `gate|up` are concatenated at load time (one GEMM each instead of 3/2).
   Mathematically identical; per-output-element accumulation order is cuBLAS's choice either way.
4. **Attention** is a plain softmax(QKᵀ/√d)V with FP32 scores and FP32 accumulation, FP16 output (no
   flash/online softmax yet). Max subtraction is exact; `__expf` is used (≈2 ulp FP32, far below FP16 rounding).
5. **RoPE tables** are computed in f64 and rounded once to f32 (HF: f32 throughout). Difference ~1e-7.
6. **LM head** outputs FP32 logits directly from the FP16 GEMM (HF-FP32 has FP32 weights; we cannot).
7. **Tied embeddings:** the embedding matrix doubles as the LM head (`tie_word_embeddings = true` for this model);
   untied checkpoints load a separate `lm_head.weight`.
8. **Sampling** runs on the host over the FP32 logits (513 KB per step over PCIe, ~30 µs).

## 5. Build / dependency notes

- `cudarc 0.19` with `driver, cublas, nvrtc, f16, dynamic-linking`. The `nvrtc` feature is enabled **only** because
  `CudaContext::load_module` takes the `nvrtc::Ptx` type; nothing is compiled by NVRTC — PTX comes from `nvcc` at
  build time. The safe cudarc `Gemm<f16>` only produces FP16 output, so `gpu/gemm.rs` calls `cublas::result::gemm_ex`
  directly (needed for FP32 logits).
- `ctx.disable_event_tracking()` is called: with a single stream cudarc's per-allocation events are pure overhead.
- CUDA toolkit 13.x (nvcc at `/opt/cuda` locally); `build.rs` honours `NVCC`, `CUDA_PATH`, `CUDA_HOME`. cudarc's
  version feature is pinned to `cuda-13040` because 0.19 rejects newer minors (13.5) under
  `cuda-version-from-build-system`; the driver API is backward compatible.
- Rust toolchain pinned in `rust-toolchain.toml`. Clippy runs with `-D warnings` in `make ci`.
- GPU tests (`engine/tests/kernels.rs`, `weights.rs`) need `--features gpu-tests`; plain `cargo test` runs CPU tests only.

## 6. Verification plan

**Forward-pass test thresholds (`tests/forward.rs`, synthetic model).** The tiny config of `tests/weights.rs`
(hidden 64, 2 layers, 4 q / 2 kv heads of 16, inter 128, vocab 100) with random BF16 weights (linear `U(±√(3/in))`,
norms `U(0.5, 1.5)`, embedding `U(±1)`; logit rms ≈ 5) is run on the GPU and compared with an f64 CPU forward
written in the test, in two flavours: exact, and one that rounds to FP16 at every point where the GPU stores an FP16
activation. Metric: max |logit error| over all rows.

- The error is quantised by single FP16 rounding flips. One ulp of a hidden value `x_k` (2⁻¹⁰·|x_k|) moves a logit by
  ≈ 2⁻¹⁰ · (|x_k|/rms(x)) · w_k · |E_vk| ≲ 2⁻¹⁰ · 3 · 1.5 · 1 ≈ 4e-3; a few such flips across 2 layers give ~1–2e-2.
  Any change of accumulation order (cuBLAS picks different algorithms for T = 1 and T = 24, f32 vs f64 sums) flips
  some roundings, so the FP16-emulating reference, GPU decode-vs-prefill and the exact reference all sit at the
  *same* floor; emulation does not buy a tighter bound.
- Measured over 30 random models × 24 tokens: worst vs FP16-emulated 2.14e-2, vs exact 2.40e-2, decode-vs-prefill
  1.99e-2. **`TOL = 5e-2`** (≈ 2× the worst of 30, ≈ 1% of logit rms) for every comparison that crosses an FP16
  rounding: GPU vs either reference, decode vs prefill, chunked prefill + decode vs reference, truncate + recompute.
- Power check, asserted in the test: references with a plausible bug differ from the GPU by ≥ 10×TOL; the smallest
  over the 30 models were 13.6 (GQA `h % n_kv` instead of `h / group`), 10.4 (interleaved instead of rotate_half RoPE)
  and 10.3 (query `i` also sees key `i+1`). A uniform position shift is *not* detectable (RoPE is relative), by design.
- `Logits::Last` vs the last row of `Logits::All` for the same tokens differ only in the FP32-output head GEMM
  (layers run with the same T, final norm is per row): bound 2·γ₆₃·Σ|aᵢbᵢ| ≈ 3e-4, measured 3.3e-6,
  **`TOL_HEAD_ONLY = 1e-3`**. A reset followed by the same call must be bit-identical.
- Top-1 must agree with the exact reference at every position whose reference top-2 margin exceeds 2·TOL.

**Real-model logit test (S1-06/S1-07, not yet built).**

- `ref/dump_logits.py`: HF `LlamaForCausalLM`, FP32, eager attention → prompt ids, prefill logits (all positions),
  greedy tokens (48 steps), decode logits for the first 16 steps (teacher-forced). Also an HF-**FP16** run, giving the
  intrinsic FP16-vs-FP32 error floor.
- `tests/logits.rs` thresholds are *calibrated from that floor* rather than guessed: max abs err ≤ 2× the HF-FP16 floor
  and ≤ 1.0 absolute, mean abs err ≤ 0.05; top-1 agreement 100% on decode steps and ≥ 99% over prefill positions
  (excluding positions whose reference top-2 margin < 0.05, reported and capped at 1%); ≥ 32 greedy tokens identical.
- Roofline: decode bytes = `Weights::decode_bytes()` (all layer matrices + norms + tied head ≈ 2.47 GB) / 504 GB/s
  ≈ 4.9 ms/token ≈ 204 tok/s upper bound (KV reads add < 1% at short context).

## 7. Status

| step | state |
|---|---|
| gpu wrappers, build.rs | done |
| 8 kernels + CPU-reference tests, GEMM test | done, 16/16 passing |
| config, RoPE table, weight loader | done; loader tested on a synthetic BF16 checkpoint (exact fused-layout check) |
| host sampling (greedy, temperature, top-p) | done; seeded, partial-select nucleus, 12 CPU tests |
| real Llama 3.2 1B weights | **missing**: gated repo, backlog `ENV-weights` |
| workspace (activation buffers, ids upload, logits D2H into a reused host buffer) | done (S1-01); CPU size tests + GPU alloc/round-trip test |
| KV cache (`KvCache`) | done (S1-02); `tests/kv.rs` (6 GPU tests: exact placement at several `pos0`, overflow errors, attention via layer view) |
| forward pass (`Llama::forward`, prefill / chunked prefill / decode) | done (S1-03); `tests/forward.rs` (6 GPU tests vs f64 CPU reference on a synthetic model, thresholds §6) |
| CLI `kvasir generate` (streamed output, EOS list, prefill/decode tok/s, `--json`) | done (S1-05); `tests/cli.rs` (6 GPU tests running the binary on a synthetic model + in-test WordLevel tokenizer) |
| ref script, logit test, perf report | ordered in `.agent/backlog.yaml` (S1-06 … S1-08) |
