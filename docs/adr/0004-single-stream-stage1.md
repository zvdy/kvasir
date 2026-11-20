# 4. One CUDA context and one stream in Stage 1

Status: accepted

## Context

Batch-1 generation has no independent work to overlap. cudarc's per-allocation event tracking costs time on every
launch.

## Decision

`Gpu` owns one context and one stream; event tracking is disabled. All kernels and cuBLAS calls go to that stream.

## Consequences

No cross-stream hazards to reason about. Continuous batching (Stage 2+) will revisit this with a new ADR.
