# 6. GitHub is the CPU gate; GPU results are local evidence

Status: accepted

## Context

Agents need a gate they cannot skip. GitHub-hosted runners have no GPU. A self-hosted GPU runner on a public repo would
let untrusted PR code run on a maintainer's machine, and paid GPU runners are T4 (sm_75), which the sm_89 PTX from
ADR 0005 does not target.

## Decision

GitHub CI runs everything that needs no GPU: fmt, clippy (with and without `gpu-tests`), CPU tests, a release build of the
GPU test binaries (nvcc compiles every kernel), docs, cargo-deny and actionlint, summarized by one `ci-pass` job.
Branch protection on `main` requires `ci-pass`, `dco` and `agent-pr`.

GPU tests run on the maintainer's RTX 4070 via `make ci-local`. Agent PRs must paste the `make test-gpu` summary into
their run log; the `agent-pr` check (`scripts/check-agent-pr.sh`) rejects PRs without it, without a CHANGELOG entry, or
with the backlog task still `todo`.

## Consequences

Nothing merges with a red CPU gate. GPU correctness relies on recorded evidence, not on re-execution, so reviewers
should spot-check GPU-heavy PRs with `make ci-local`. Revisit when an isolated GPU runner (ephemeral VM, sm_89) is
available.
