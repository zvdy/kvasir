# 2. Hand-written kernels except GEMM (cuBLAS)

Status: accepted; amended by 0007 (hand-written GEMV for decode)

## Context

The project's point is to write the inference stack from scratch, but competitive FP16 GEMM needs tensor-core tiling
that is a project in itself.

## Decision

All non-GEMM ops are hand-written CUDA kernels. Linear layers use `cublasGemmEx` (FP16 in, FP32 accumulate) called
directly via `cudarc::cublas::result` so FP32 outputs (logits) are possible. No ML framework is a dependency.

## Consequences

Linear-layer performance is near-roofline from day one. Replacing cuBLAS with a custom GEMM is allowed later only with a
benchmark showing parity.
