# kvasir — design notes (Stage 1: Llama 3.2 1B, batch 1)

Target: one RTX 4070 (12 GB, sm_89, ~504 GB/s). Rust + hand-written CUDA kernels; cuBLAS only for GEMMs with T > 1
(decode GEMVs are hand-written, ADR 0007).

## 1. Module boundaries

```
engine/
  build.rs            nvcc -ptx -arch=sm_89 for kernels/*.cu -> $OUT_DIR/*.ptx (embedded with include_str!)
  kernels/            embedding.cu rmsnorm.cu rope.cu attention.cu silu_mul.cu elementwise.cu gemv.cu common.cuh
  src/error.rs        KvasirError (+ ensure_shape! macro). No unwrap/expect outside tests and main.
  src/gpu/
    device.rs         Gpu { ctx, stream, blas, kernels } — one context, ONE stream (stage 1)
    buffer.rs         DeviceTensor<T>: owned CudaSlice + logical shape, deliberately not Clone
    kernels.rs        one shape-checked launch wrapper per kernel, as methods on Gpu
    gemm.rs           linear_f16 / linear_f32: cublasGemmEx, FP16 in, FP32 accumulate (T > 1);
                      gemv_f16 / gemv_f32 / gemv_add / gemv_silu_mul: decode GEMV, optional fused RMSNorm (§3)
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
engine/benches/       kernels.rs: per-kernel microbenchmarks vs bench/kernels-baseline.json;
                      forward.rs: synthetic full-size 1B prefill/decode vs bench/forward-baseline.json (`make bench`)
tests/ , ref/         logit-matching test and HF reference dumper        [not written yet]
```

Ownership: device memory is owned by exactly three long-lived structs — `Weights`, `KvCache`,
`Workspace`. Kernel wrappers take borrowed `CudaView`/`CudaViewMut`, never allocate. The generation
loop therefore performs no device allocation (the logits D2H copy goes into a reused host `Vec`).
The attention split-K scratch is per-call working memory, so `Workspace` owns it (sized from `n_q`, `head_dim`);
`KvCache` holds only sequence state (K/V, RoPE tables, `len`), and `KvCache::attention` takes the scratch as an
argument exactly like the kernel wrapper.

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
FP32, `attn_scratch [attention_scratch_len(n_q, head_dim)]` FP32 (zeroed at allocation, one buffer for all layers,
§3), plus host buffers for the ids and the logits copy. A call with `t` tokens uses the leading `t` rows of each
buffer. Llama 3.2 1B: 70 KiB per token of FP16 activations + 501 KiB per head row + 1.08 MB attention scratch.

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

**Prefill vs decode.** Same layer structure; `T = prompt_len, pos0 = 0` vs `T = 1, pos0 = cur_len`. At `T = 1` the
projections are the fused GEMVs of §3 instead of cuBLAS + separate norm / SwiGLU / add kernels (ADR 0007).
Query `i` attends to keys `[0, pos0+i]`. Only the *last* row goes through the final norm and LM head during
generation (the logit-matching test asks for all rows).

**Forward pass (`model::Llama`).** `Llama` owns `Weights`, `KvCache` and `Workspace` and holds no other device
memory. `forward(gpu, ids, Logits::Last | Logits::All)` runs `ids` at positions `len..len+T` and returns a host
slice of FP32 logits (`[1, vocab]` or `[T, vocab]`, borrowed from the workspace's reused host buffer). Per layer:
`rmsnorm → W_qkv → rope_inplace → cache.append → cache.attention → W_o → hidden += proj → rmsnorm → W_gu →
silu_mul → W_down → hidden += proj`; then `cache.advance(T)`, the final norm on the last `T_head` rows only (written
to the leading rows of `normed`) and the LM head GEMM into `logits`. At `T = 1` a layer is `gemv_f16(rmsnorm(hidden)) →
qkv → rope → append → attention → gemv_add(W_o, attn) into hidden → gemv_silu_mul(rmsnorm(hidden)) → act →
gemv_add(W_down, act) into hidden` (7 launches; `normed`, `proj` and `gate_up` are not touched), and whenever
`T_head = 1` (decode, or prefill with `Logits::Last`) the head is `gemv_f32` with the final norm fused, reading the
last `hidden` row directly. All views are leading-row slices
(`DeviceTensor::row_view`); nothing is allocated per call. Inputs (`1 ≤ T ≤ max_tokens`, `T_head ≤ max_head_rows`,
`len + T ≤ max_seq`, ids `< vocab`) are validated before any device work, so a rejected call leaves the sequence
unchanged; `len` only advances after the last layer. `Llama::new` rejects `max_tokens > max_seq`, head shapes
without an attention variant (`head_dim ∉ {64, 128}` or GQA group `∉ {1, 2, 4, 8}`) and `max_seq >
max_position_embeddings` (in `KvCache::from_config`), so limits surface at construction instead of mid-generation.
Since A-01 the kernels have no kv-length cap of their own: `Llama::max_seq_limit(head_dim)` = `i32::MAX` (`max_seq`
is an i32 kernel argument; cache offsets are 64-bit); the KV cache size is the practical bound (32 KiB per token for
1B). The attention wrapper also returns a `Shape` error if `qkv`/`kc`/`vc` are not 16-byte aligned or `out`/`scratch`
not 4-byte aligned; every view `forward` passes is aligned by construction (leading-row slices of fresh allocations;
per-layer cache offsets are multiples of `max_seq·head_dim·2 B`). `reset`/`truncate` rewind the cache.

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

## 3. Kernels (validated against CPU before use)

| kernel | grid × block | notes |
|---|---|---|
| `embedding_gather` | T × 256 | ids range-checked on host before upload |
| `rmsnorm` | rows × 256 | FP32 sum of squares; block reduce via warp shuffles |
| `rope_inplace` | T × 128 | one thread per (head, pair) |
| `kv_append` | T × 128 | |
| `attention_decode_{hd}_{g}` | (8, splits) × 128 | `T = 1`, flash-decoding; see below |
| `attention_prefill_{hd}_{g}` | (⌈T/(64/g)⌉, 8) × 128 | `T > 1`, FA2-style on tensor cores; see below |
| `silu_mul` | grid-stride | |
| `add_inplace` | grid-stride | residual add |
| `bf16_to_f16` | grid-stride | counts values outside FP16 range |
| `gemv_{f16,f32,add,silu_mul}` | ⌈rows/8⌉ × 128 | `T = 1` projections; see below |

**Decode GEMV** (P-01, ADR 0007). `y[out] = W[out, in]·x[in]` with `W` in the HF `[out, in]` layout, FP16 in, FP32
accumulate. One warp owns 2 output rows (`gemv_silu_mul`: one gate row and its up row), 4 warps per block. Lane `l`
reads the 16-byte chunks `l, l+32, …` of each row with `ld.global.nc.L1::no_allocate` (8 chunks per row in flight);
the matching x chunk comes through `__ldg` (L1-resident, shared by both rows), then two FP32 FMAs per half2 and a
5-step shuffle reduction. Out-of-range rows of the last warp re-read the last row and are not stored. Epilogues:
`f16` (round once), `f32` (logits), `add` (`y = half(y + half(W·x))`, the same two roundings as `proj` + `add_inplace`)
and `silu_mul` (`act = half(silu(half(g))·half(u))`, as `silu_mul` read the FP16 `gate_up`). Optional prologue: with a
norm weight the kernel reads the raw hidden row and normalises each x chunk as `half(x·inv·w)` exactly like `rmsnorm`;
each warp computes `inv` itself from the x chunks it already loaded (`in ≤ 2048`; a second L1 pass otherwise), after
its weight loads are issued, so the fused norm costs ≈ 0.3 µs instead of a 3.2 µs launch. Wrappers check `in % 8 = 0`
(else `Shape`), lengths, and 16-byte alignment of `x`, `W` and the norm weight; `Llama::new` rejects configs whose
hidden / q_dim / intermediate are not multiples of 8. Tuning: rows per warp 1/2/4, 2–8 warps per block, unroll 4/8/16
and `.cs` / `.nc` / `L2::128B|256B` load variants all measured within ±1% — the kernel is at the DRAM streaming rate.

**Attention** (A-01). One wrapper, `Gpu::attention(qkv, kc, vc, out, scratch, t, pos0, n_q, n_kv, head_dim, max_seq)`,
dispatches on `t`. Variants are compiled for `head_dim ∈ {64, 128}` × GQA group `g = n_q/n_kv ∈ {1, 2, 4, 8}`; other
shapes are a `Shape` error. `qkv`, `kc`, `vc` must be 16-byte aligned (true for allocations and per-layer cache
slices). There is no kv-length cap beyond `max_seq` (tested to 40k). Scores are FP32 in the exp2
domain (q·k scaled by `log2 e/√d`), with online (running max/sum) softmax.

- *Decode:* one block per (KV head, context split), so each K/V row is loaded once for all `g` query heads. Eight
  lanes read one 128-byte row with 16-byte `ld.global.cs` loads (a warp covers 4 rows per load, 4 loads in flight per
  lane, next round prefetched); partial dot products are reduced with 3 shuffles. Splits: `chunk = max(256,
  ⌈len/128⌉)` keys, `splits = ⌈len/chunk⌉` (≤ 128). With one split the block writes `out` directly. Otherwise each block
  writes `(m, l, acc[hd])` to `scratch` and bumps a per-KV-head counter; the last block to arrive merges all splits and
  resets the counter to 0 (last-block-reduces, no second launch). Chunk 256 beat 64/128 at every benched context:
  fewer, longer streams per block lose less to ramp/tail than more blocks gain in parallelism.
- *Scratch:* `scratch: &mut CudaViewMut<f32>` with at least `kernels::attention_scratch_len(n_q, head_dim)` =
  `n_q + 128·n_q·(head_dim+2)` floats (1.08 MB for Llama 3.2 1B, one buffer shared by all layers). Layout: `n_q` u32
  counter words, then partials `[split, n_q, head_dim+2]`. It must be **zeroed once at allocation** (`alloc_zeros`);
  every call leaves the counters at zero. Prefill does not touch it. Owned by `Workspace` (`attn_scratch`, I-01) —
  the wrapper does not allocate.
- *Prefill:* block = 64/g query tokens × g heads of one KV head = 64 query rows, 4 warps × 16 rows. Q fragments live
  in registers; K/V tiles (64 keys for hd 64, 32 for hd 128) are double-buffered in smem with `cp.async` (zero-filled
  past the causal end, rows padded by 16 B for conflict-free access). S = QKᵀ and O += PV use
  `mma.sync.m16n8k16` (FP16 in, FP32 accumulate); S accumulators are re-packed in registers as the A operand of PV and
  V fragments come from `ldmatrix.trans`. Causal mask per element (`key ≤ pos0 + token`); query tiles are scheduled
  longest-first. No split over keys, so prefill with a long prefix and few new tokens under-fills the GPU.

Unit-test results (`cargo test --test kernels`, 31 tests): all outputs are within one FP16 output rounding
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
4. **Attention** is softmax(QKᵀ/√d)V with FP32 scores and accumulation, FP16 output, computed with an online
   softmax in base 2 (`exp2f`, ≈2 ulp). Decode is FP32 throughout. Prefill rounds P to FP16 for the tensor-core P·V
   (as FA2 does) and sums the *rounded* P into the denominator so the weights still sum to one. All attention tests
   stay within the unchanged `2e-3 rel + 1e-3 abs` (max error 6.6e-4, same as the old kernel).
5. **RoPE tables** are computed in f64 and rounded once to f32 (HF: f32 throughout). Difference ~1e-7.
6. **LM head** outputs FP32 logits directly from the FP16 GEMM (HF-FP32 has FP32 weights; we cannot).
   At `T = 1` all projections use the GEMV (FP32 accumulate in a fixed lane/shuffle order) instead of cuBLAS, so decode
   and prefill differ in summation order; the GEMV epilogues keep exactly the FP16 roundings of the separate kernels.
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

**Forward-pass test thresholds (`tests/forward.rs`, synthetic model).** A tiny config (hidden 256, 2 layers, 4 q /
2 kv heads of 64, inter 128, vocab 100) with random BF16 weights (linear `U(±√(3/in))`, norms `U(0.5, 1.5)`,
embedding `U(±0.5)`; logit rms ≈ 5) is run on the GPU and compared with an f64 CPU forward
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
- I-01 moved the model from heads of 16 (hidden 64, embedding `U(±1)`) to heads of 64, because the A-01 attention
  kernels only exist for head_dim 64/128. Logit and error both scale with `√hidden·|E|`, so the embedding range was
  halved with the 4× wider hidden to keep logit rms ≈ 5 and TOL at ≈ 1% of it; tolerances are unchanged. Re-measured
  over 30 random models × 24 tokens: worst vs FP16-emulated 2.67e-2, vs exact 2.74e-2, decode-vs-prefill 2.51e-2,
  head-only 1.1e-5 (TOL ≈ 1.8× the worst of 30); smallest bug-reference distances 321 / 213 / 187 × TOL (GQA, RoPE,
  causal). The other measurements in this list are from the head_dim-16 model.
- P-01 (decode GEMV, different summation order than the prefill cuBLAS GEMM): same 30-model sweep, decode-vs-prefill
  2.51e-2 → 2.82e-2, decode vs FP16-emulated 1.93e-2 → 2.18e-2, vs exact 2.60e-2 → 2.60e-2 (TOL ≈ 1.8× the worst;
  tolerances unchanged).
- Power check, asserted in the test: references with a plausible bug differ from the GPU by ≥ 10×TOL; the smallest
  over the 30 models were 13.6 (GQA `h % n_kv` instead of `h / group`), 10.4 (interleaved instead of rotate_half RoPE)
  and 10.3 (query `i` also sees key `i+1`). A uniform position shift is *not* detectable (RoPE is relative), by design.
- `Logits::Last` vs the last row of `Logits::All` for the same tokens differ only in the FP32-output head GEMM
  (layers run with the same T, final norm is per row): bound 2·γ₆₃·Σ|aᵢbᵢ| ≈ 3e-4, measured 3.3e-6,
  **`TOL_HEAD_ONLY = 1e-3`**. A reset followed by the same call must be bit-identical.
- Top-1 must agree with the exact reference at every position whose reference top-2 margin exceeds 2·TOL.

**Real-model logit test (S1-06/S1-07).**

- `ref/dump_logits.py`: HF `LlamaForCausalLM`, FP32, eager attention → prompt ids, prefill logits (all positions),
  greedy tokens (48 steps), decode logits for the first 16 steps (teacher-forced). Also an HF-**FP16** run, giving the
  intrinsic FP16-vs-FP32 error floor.
- `make ref` writes `ref/out/` (CPU torch, deps locked in `ref/requirements.txt`; skips with a message when the weights
  are absent). Raw little-endian files, no header, row-major: `prompt_ids.u32` `[T]`, `greedy_ids.u32` `[48]`,
  `prefill_logits.f32` `[T, V]`, `decode_logits.f32` `[16, V]`, and the same two logit files with suffix `_hf16`
  (FP16 run, upcast to f32). `greedy_ids[0]` is the argmax of the last prefill row; `decode_logits[k]` is the output at
  position `T + k` after feeding `greedy_ids[k]`. The FP16 run is teacher-forced on the FP32 greedy tokens.
  `meta.json` (`format_version: 1`) holds shapes/dtypes per file, prompt, token ids, checkpoint sha256 and revision,
  seed, torch/transformers versions, and the HF-FP16 floor (`hf16_floor.{prefill,decode}`: max/mean abs err, top-1).
- `tests/logits.rs` thresholds are *calibrated from that floor* rather than guessed: max abs err ≤ 2× the HF-FP16 floor
  and ≤ 1.0 absolute, mean abs err ≤ 0.05; top-1 agreement 100% on decode steps and ≥ 99% over prefill positions
  (excluding positions whose reference top-2 margin < 0.05, reported and capped at 1%); ≥ 32 greedy tokens identical.
- `tests/logits.rs` (feature `model-tests`, `make test-model`, not in `ci-local`) reads `ref/out/` and
  `models/Llama-3.2-1B/` and fails with a pointer to `make ref` when either is missing. Max-abs threshold per part =
  min(2 × that part's floor, 1.0). Weights: `model.safetensors` from the mirror `unsloth/Llama-3.2-1B`, sha256
  `68a2e4be…6809a`, identical to `meta-llama/Llama-3.2-1B`'s; the mirror's `tokenizer.json` differs in bytes from
  Meta's (unverified), which does not affect the test because it feeds the token ids from `meta.json`.
- **Measured (2026-11-23, 35-token default prompt, torch 2.14.1+cpu / transformers 5.22.2).** HF-FP16 floor: prefill
  max 0.0256 / mean 0.00254, decode max 0.0191 / mean 0.00265, top-1 100% both. kvasir: prefill max **0.0210**
  (≤ 0.0512) / mean **0.00225**; decode (teacher-forced, 16 steps) max **0.0168** (≤ 0.0381) / mean **0.00211**;
  prefill top-1 **29/29** on non-excluded positions (35/35 including excluded). Failing, all on reference near-ties:
  decode top-1 **15/16** (step 1: reference margin 0.0027 between ids 4358 and 7664, kvasir has 7664 ahead by
  0.0005; HF-FP16 happens to keep 4358); free-running greedy identical prefix **2/48** (the same tie, then a
  different but coherent continuation); prefill positions with margin < 0.05 **6/35** (17%, cap 1%). The logit
  errors are below the FP16 floor, so the top-1/greedy/exclusion criteria are not attainable for this prompt by any
  FP16 engine without luck: a 0.0027 margin is ≈ 1/7 of the floor. **S1-07 is blocked** on a human decision about
  these criteria (e.g. exempt decode steps / greedy positions whose reference margin is below the floor, or define the
  exclusion cap in positions rather than 1% of 35); tolerances were not changed.
- Roofline: decode bytes = `Weights::decode_bytes()` (all layer matrices + norms + tied head ≈ 2.47 GB) / 504 GB/s
  ≈ 4.9 ms/token ≈ 204 tok/s upper bound (KV reads add < 1% at short context).

## 7. Status

| step | state |
|---|---|
| gpu wrappers, build.rs | done |
| 8 kernels + CPU-reference tests, GEMM test | done, 25/25 passing |
| config, RoPE table, weight loader | done; loader tested on a synthetic BF16 checkpoint (exact fused-layout check) |
| host sampling (greedy, temperature, top-p) | done; seeded, partial-select nucleus, 12 CPU tests |
| real Llama 3.2 1B weights | present (`ENV-weights` done): mirror `unsloth/Llama-3.2-1B`, `model.safetensors` sha256 `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` = `meta-llama/Llama-3.2-1B`'s; mirror tokenizer.json unverified (§6) |
| workspace (activation buffers, ids upload, logits D2H into a reused host buffer) | done (S1-01); CPU size tests + GPU alloc/round-trip test |
| KV cache (`KvCache`) | done (S1-02); `tests/kv.rs` (6 GPU tests: exact placement at several `pos0`, overflow errors, attention via layer view) |
| forward pass (`Llama::forward`, prefill / chunked prefill / decode) | done (S1-03); `tests/forward.rs` (6 GPU tests vs f64 CPU reference on a synthetic model, thresholds §6) |
| kernel microbenchmarks + stored baseline (`make bench`, H-01) | done; numbers in §8 |
| flash-decoding / FA2 prefill attention (A-01) | done; decode ctx 2048 at 60% of DRAM bandwidth, prefill T=128 25× faster |
| attention wired into the forward pass, full-size forward bench (I-01) | done; decode 5.84 ms/token at ctx 2048 (84% of roofline), prefill 128 tok in 7.6 ms; §8 |
| hand-written decode GEMV with fused RMSNorm / SwiGLU / residual add (P-01, ADR 0007) | done; 6 CPU-reference tests (real + odd shapes, shape errors); decode 5.69 ms/token at ctx 2048 (86% of roofline, was 84%) |
| CLI `kvasir generate` (streamed output, EOS list, prefill/decode tok/s, `--json`) | done (S1-05); `tests/cli.rs` (6 GPU tests running the binary on a synthetic model + in-test WordLevel tokenizer) |
| HF reference dumper (`ref/dump_logits.py`, `make ref`) | done (S1-06); run on the real weights, HF-FP16 floor prefill 0.0256 / decode 0.0191 max abs |
| real-model logit test (`tests/logits.rs`, `make test-model`) | **blocked** (S1-07): logit error below the HF-FP16 floor, but decode top-1 15/16, greedy prefix 2/48 and 6/35 low-margin prefill positions fail on reference near-ties; §6 "Measured" |
| perf report | ordered in `.agent/backlog.yaml` (S1-08) |

## 8. Benchmarks

### Kernel microbenchmarks

`make bench` (`engine/benches/kernels.rs`, `harness = false`, needs `gpu-tests`) times every kernel at Llama 3.2 1B
dims, decode (`T=1`) and prefill (`T=128`), and fails if any median is > 10% slower than `bench/kernels-baseline.json`;
`make bench-baseline` rewrites the baseline. No criterion: CUDA events are what we need and it would be a new dependency.

Method: buffers are allocated before timing; each sample queues a ~1 ms filler GEMM, then CUDA events around `reps`
back-to-back launches (sized to ~300 µs), so host launch latency is hidden but inter-kernel gaps are included.
Median of 101 samples after warmup and a 0.5 s clock ramp. Weights and the read KV span rotate over ≥ 160 MiB of copies so
they come from DRAM, as in a real 16-layer pass; activations are deliberately L2-resident (as they are in a pass),
which is why some prefill elementwise kernels exceed the DRAM roofline. Reruns on an idle GPU agree within ~2%.

Initial baseline (RTX 4070, driver 620.70.06), µs; % = of 504 GB/s, or of the 58.3 TFLOP/s FP16 spec peak for
prefill GEMMs:

| kernel | T=1 | % | T=128 | % |
|---|---:|---:|---:|---:|
| embedding | 3.25 | 1 | 4.24 | 49 |
| rmsnorm | 3.17 | 1 | 3.57 | 59 |
| rope | 2.66 | 1 | 3.07 | 87 |
| kv_append | 1.68 | 0 | 1.84 | 56 |
| silu_mul | 1.42 | 7 | 8.72 | L2 |
| add | 1.30 | 2 | 2.46 | L2 |
| attention, ctx 512 / 2048 / 8192 (A-01) | 7.16 / 14.0 / 43.3 | 29 / 60 / 77 | 6.99 (ctx 128) | 37 (17% FP16 peak) |
| gemm qkv / o / gate_up / down | 29.7 / 20.5 / 146 / 74.8 | 84 / 81 / 92 / 89 | 46.2 / 28.3 / 187 / 119 | 60 / 64 / 78 / 61 |
| lm_head (FP32 out) | 1113 | 94 | 1394 | 83 |
| gemv qkv (+norm) / o+add / gate_up+silu (+norm) / down+add (P-01) | 29.0 (29.3) / 20.1 / 144.5 (144.9) / 74.5 | 86 (85) / 83 / 92 (92) / 89 | – | – |
| gemv lm_head (+norm), FP32 out (P-01) | 1112 (1113) | 94 (94) | – | – |

Before A-01 attention was 79.5 / 311 / 1231 µs (3% of bandwidth) at decode and 178 µs at prefill: 32 blocks on 46 SMs,
one thread per key walking 64 elements serially, 2 threads per output dim in P·V, and every KV head read by 4 blocks.

Decode at ctx 2048 now sums to ≈ 16 × (271 GEMM + 14 attention + 15 small) µs + 1113 µs ≈ 5.9 ms/token, vs the 4.9 ms
roofline. Decode attention has a ~4 µs fixed cost per call (launch gap, DRAM latency ramp, split merge; ctx 512 is
29% of bandwidth for that reason) and streams at ≈ 430 GB/s marginally. The decode GEMMs are within 6–19% of
bandwidth. The small kernels are launch-bound (~1.3–3.3 µs each, ~10 per layer) and call for fusion and CUDA graphs:
≈ 16 × 15 µs ≈ 0.25 ms/token. Remaining gap to the roofline at ctx 2048: GEMM shortfall ≈ 0.5 ms, small kernels
≈ 0.25 ms, attention ≈ 0.1 ms (its fixed cost).

### End-to-end forward (I-01)

`make bench-forward` (`engine/benches/forward.rs`; `make bench` runs it after the kernel bench) builds the real
Llama 3.2 1B shapes (hidden 2048, 16 layers, 32/8 heads of 64, inter 8192, vocab 128256, tied head, llama3 RoPE) with
seeded random weights uploaded from host pools (0.2 s) and a 4096-position cache, then times through `Llama::forward`:
prefill of 128 tokens from an empty cache, and decode at ctx 128 / 512 / 2048 (cache filled by 128-token prefill
chunks, `truncate(ctx)` before every step so the context is exact; 10 warm-up + 60 timed steps, `Logits::Last`).
*wall* = host `Instant` around the whole call (ids upload, launches, logits D2H, sync), i.e. the latency a caller
sees; *device* = CUDA events around the call while it is queued behind a filler GEMM ≥ 3× its wall time, so host
launch latency is hidden. The logits D2H (128256 FP32) and a host argmax are timed separately, not gated. Medians;
`bench/forward-baseline.json`, fails on > 10% regression of any gated median (same rule as the kernel bench, one
gate for both; reruns agree within 1%). Roofline = `Weights::decode_bytes()` 2.472 GB / 504 GB/s = 4.904 ms/token.

| case | median ms | tok/s | % of roofline |
|---|---:|---:|---:|
| prefill 128 tokens | 7.65 | 16 700 | – |
| decode ctx 128 (wall / device) | 5.67 / 5.75 | 176 | 86 / 85 |
| decode ctx 512 | 5.75 / 5.80 | 174 | 85 / 85 |
| decode ctx 2048 | 5.83 / 5.91 | 171 | 84 / 83 |
| logits D2H + sync / host argmax | 0.040 / 0.056 | | |

Wall ≤ device: the host enqueues the ≈ 180 launches of a step faster than the GPU drains them (the 145 µs gate/up
GEMMs and the 1.1 ms LM head give it slack), so host launch overhead is already off the critical path and a CUDA
graph alone would only remove inter-kernel gaps. Decode is GPU-bound.

**Decode step breakdown, ctx 2048** (temporary CUDA events after every op, behind the filler; the events add ≈ 1 µs
per op, total 6.14 ms vs 5.91 ms unprofiled), µs per token:

| op (calls) | µs | per call | % of DRAM bw |
|---|---:|---:|---:|
| gate_up GEMM (16) | 2309 | 144.3 | 92 |
| down GEMM (16) | 1199 | 74.9 | 89 |
| LM head, FP32 out (1) | 1105 | 1105 | 94 |
| qkv GEMM (16) | 484 | 30.3 | 83 |
| o GEMM (16) | 340 | 21.3 | 78 |
| attention (16) | 246 (94 at ctx 128) | 15.4 (5.9) | 54 |
| rmsnorm (33) | 182 | 5.5 | – |
| add (32) | 79 | 2.5 | – |
| rope / kv_append / silu_mul / embedding (49) | 150 | 2.5–5 | – |
| logits D2H (1) | 51 | 51 | – |

Gap to roofline at ctx 2048, wall 5.84 − 4.90 = 0.94 ms. Top bottlenecks, largest first:

1. **cuBLAS GEMV efficiency at T = 1: ≈ 0.54 ms.** The five weight GEMMs take 5.44 ms against 4.90 ms of weight
   bytes; o (78%) and qkv (83%) are worst, gate_up/down lose 0.32 ms between them. A hand-written split-K FP16 GEMV
   at ≥ 95% of bandwidth (and fusing silu_mul into the gate_up epilogue, the residual add into o/down) would recover
   ≈ 0.35–0.45 ms (≈ +7% tok/s).
2. **Small kernels: ≈ 0.3 ms** (≈ 130 launches of 2.5–5.5 µs, ≈ 0.4 ms with event overhead), each latency-bound on
   a 4 KB row. Fusions: residual add + next RMSNorm (removes 32 launches and the `proj` round trip), rope +
   kv_append (16), silu_mul into the GEMV (16); then a CUDA graph for the remaining inter-kernel gaps. Estimated
   gain ≈ 0.2–0.25 ms.
3. **Attention fixed cost and the LM head: ≈ 0.1 + 0.1 ms.** Attention at ctx 2048 moves 67 MB of KV (133 µs at
   roofline) in 246 µs; ≈ 5 µs of each call is fixed (94 µs total at ctx 128), so it grows with context (≈ 20% of the
   step at 8k). The LM head is at 94% of bandwidth but is 19% of the step; the remaining lever there is bytes
   (quantised head) and returning the argmax/sampled id from the device instead of 513 KB of logits plus a 56 µs host
   argmax (≈ 0.1 ms of host-serial time per generated token in `kvasir generate`).

### Decode GEMV (P-01)

Same benches, `T = 1` projections on the hand-written GEMV of §3 (ADR 0007). Per matrix, µs and % of 504 GB/s
(weights + x + y bytes):

| matrix | cuBLAS `T=1` | GEMV | GEMV fused (as used) |
|---|---:|---:|---:|
| qkv [3072, 2048] | 29.70 (84%) | 28.95 (86%) | + RMSNorm prologue: 29.29 (85%) |
| o [2048, 2048] | 20.55 (81%) | – | + residual add: 20.14 (83%) |
| gate_up [16384, 2048] | 145.41 (92%) | + SwiGLU: 144.47 (92%) | + RMSNorm: 144.92 (92%) |
| down [2048, 8192] | 75.26 (89%) | – | + residual add: 74.46 (89%) |
| LM head [128256, 2048], FP32 | 1112.1 (94%) | 1112.1 (94%) | + final RMSNorm: 1113.1 (94%) |

The GEMV is at the hardware streaming rate: over plain GEMVs of 1024–16384 rows × 2048 (4–67 MB) the time is
`2.1 µs + bytes / 470.6 GB/s` to within 0.3 µs, i.e. 93% of spec bandwidth in steady state plus a fixed ≈ 2.1 µs per
launch (launch gap, DRAM ramp, drain). No launch shape or load flavour moves it (§3). Hence the ≥ 90% target is met by
the large matrices (gate_up 92%, LM head 94%, down 89–90%) and is out of reach for o (8 MB: best case 2.1 + 17.8 µs =
84%) and qkv (12.6 MB: 87%) until kernel boundaries overlap. The gain over cuBLAS is mostly the fusions: per layer
5 launches fewer (2 rmsnorm, silu_mul, 2 add; ≈ 13 µs) and the `normed`/`gate_up`/`proj` round trips.

Forward bench (`make bench-forward`), median ms/token, before = I-01 baseline, after = P-01 (two reruns within 0.1%):

| case | before ms (tok/s, % roofline) | after ms (tok/s, % roofline) | Δ |
|---|---:|---:|---:|
| decode ctx 128 wall / device | 5.674 / 5.753 (176, 86%) | 5.515 / 5.534 (181, 89%) | −2.8% / −3.8% |
| decode ctx 512 wall / device | 5.748 / 5.800 (174, 85%) | 5.560 / 5.580 (180, 88%) | −3.3% / −3.8% |
| decode ctx 2048 wall / device | 5.833 / 5.911 (171, 84%) | 5.669 / 5.688 (176, 87%) | −2.8% / −3.8% |
| prefill 128 tokens | 7.654 | 7.653 | 0 (cuBLAS path unchanged) |

Gap to roofline at ctx 2048, wall 5.67 − 4.90 = 0.77 ms, now ≈ 114 launches per token:

1. **DRAM efficiency ceiling: ≈ 0.35 ms.** Streaming 2.47 GB at the measured 470.6 GB/s instead of 504 costs
   4.90 × (504/470.6 − 1). Not recoverable at FP16; only fewer bytes (FP8/INT8 weights, quantised head) move it.
2. **Kernel boundaries: ≈ 0.14 ms GEMV + ≈ 0.15 ms small kernels.** 65 GEMV launches × 2.1 µs fixed, plus per layer
   rope (2.7 µs) + kv_append (1.7 µs) + attention's ≈ 5 µs fixed cost. Next steps: programmatic dependent launch
   (`cuLaunchKernelEx` + `griddepcontrol`) so each GEMV issues its weight loads while the previous kernel drains (the
   weights do not depend on it), rope + kv_append fused into the qkv GEMV epilogue (pair rows `j`, `j+32` of a head in
   one warp), then a CUDA graph for the step.
3. **Attention KV bytes: ≈ 0.13 ms at ctx 2048** (67 MB, not in the weight roofline); grows with context.

