# 2026-11-23 ci-pipeline: GitHub CI as the agent gate, parallel agent harness
## doctor
gpu=ok (NVIDIA GeForce RTX 4070, 620.70.06); weights=no
## baseline
red on main (rand 0.10, fixed in agent/fix-2026-11-23 / #4, which this branch is based on)
## changes
- .github/actions/setup/action.yml: shared toolchain/CUDA stub/cache setup
- .github/workflows/ci.yml: parallel fmt, clippy, test, build, doc, deny, actionlint + `ci-pass` aggregator
- .github/workflows/agent-pr.yml, scripts/check-agent-pr.sh: run log + GPU evidence + CHANGELOG + backlog status check
- Makefile: `build-gpu-tests` (in `ci`), `check-agent-pr`
- AGENTS.md, .claude/skills/maintain/SKILL.md, .claude/agents/reviewer.md: gates per ADR 0006, `/maintain parallel`
- .gitignore: .claude/worktrees/ (subagent worktrees)
- .claude/settings.json: allow gh pr checks / run view, git worktree, merge origin/main, flock'd make/cargo
- .github/pull_request_template.md: GPU test output block, required checks item
- .agent/backlog.yaml: S1-02 files += engine/src/lib.rs; new S1-09 llama.cpp/vLLM comparison
- docs/adr/0006-github-cpu-gate-local-gpu-evidence.md
- CHANGELOG.md
## gate
make ci: exit 0 (fmt-check, lint, test-cpu, build-gpu-tests, deny, doc)
make test-gpu:
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.33s
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
make check-agent-pr before the run log existed: exit 2 ("missing run log"), as intended
## review
harness/CI only; no engine code, tolerances or dependencies touched
## follow-ups
- after merge: branch protection on main requiring ci-pass, dco, agent-pr
- clean up unused-dependency warnings (clap, rand, tokenizers) as S1-04/S1-05 start using them
