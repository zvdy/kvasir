# 5. Kernels compiled to PTX by nvcc at build time

Status: accepted

## Context

Kernels can be compiled at runtime by NVRTC or ahead of time by nvcc.

## Decision

`engine/build.rs` compiles `kernels/*.cu` with `nvcc -ptx -arch=sm_89 -O3 -lineinfo`; the PTX is embedded with
`include_str!`. The cudarc `nvrtc` feature is enabled only for the `Ptx` type. cudarc's CUDA version is pinned by
feature (`cuda-13040`) rather than detected, so builds do not depend on the exact installed minor version.

## Consequences

Kernel compile errors surface at `cargo build`. Building needs nvcc (CI uses a CUDA devel container) but not a GPU.
Supporting other architectures means adding `-arch` targets in build.rs.
