# Contributing to kvasir

Thanks for helping. kvasir is maintained by humans and by coding agents following the same rules in
[AGENTS.md](AGENTS.md). Read it and [DESIGN.md](DESIGN.md) before sending a change.

## Setup

- Rust toolchain from `rust-toolchain.toml` (rustup installs it automatically)
- CUDA toolkit 13.x with `nvcc`, an NVIDIA GPU (sm_89 is the current target)
- `cargo install cargo-deny`
- `make doctor` shows what is missing

## Workflow

1. Pick an item from `.agent/backlog.yaml` or open an issue first for anything larger than a bug fix.
2. Branch from `main`. Keep one logical change per commit.
3. Run `make fmt` then `make ci-local`. GitHub CI has no GPU, so paste the GPU test output into the PR.
4. Fill in the pull request template.

## Rules that reviews enforce

- Kernels land together with a CPU-reference test. Tolerances are justified in DESIGN.md and never loosened to make a
  test pass.
- GPU tests are registered with `required-features = ["gpu-tests"]`.
- Minimal code comments; explanations go to DESIGN.md or an ADR in `docs/adr/`.
- Update `CHANGELOG.md` under `Unreleased`.

## Developer Certificate of Origin

All commits must be signed off (`git commit -s`), certifying the [DCO](https://developercertificate.org/).
The `dco` workflow rejects unsigned commits.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
