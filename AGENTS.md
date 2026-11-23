# Agent guide for kvasir

This file is the runbook for any coding agent (Claude Code, Codex, etc.) working in this repo.
Read DESIGN.md before touching `engine/`. Architecture rationale lives in `docs/adr/`.

## Commands

| command | what it does |
|---|---|
| `make doctor` | environment report, one `key=value` per line; non-zero if a required tool is missing |
| `make ci` | fmt-check, clippy (`-D warnings`), CPU tests, GPU-test build, cargo-deny, docs. Same as GitHub Actions |
| `make ci-local` | `ci` + GPU tests. **The gate every change must pass** |
| `make fmt` | the only target that rewrites files |
| `make kernels` | GPU kernel tests with error magnitudes printed |
| `make check-agent-pr` | the `agent-pr` GitHub check, run locally against `origin/main` |

## Gates (ADR 0006)

GitHub has no GPU. A change is accepted only when **both** hold:

1. GitHub: the `ci-pass`, `dco` and `agent-pr` checks are green on the PR (`gh pr checks <n> --watch`). These are
   required by branch protection on `main`; nothing merges red.
2. Local GPU evidence: `make ci-local` was run on a GPU machine and its `make test-gpu` summary lines
   (`test result: ok. ...`) are pasted into the run log's `## gate` section. `agent-pr` fails without them.

GPU tests live in `engine/tests/*.rs` and are gated by the `gpu-tests` feature (`required-features` in
`engine/Cargo.toml`). Any new test that needs a GPU must be registered the same way.

## Invariants (do not break)

- No `unwrap`/`expect` outside tests and `src/bin/`. Errors go through `KvasirError` / `ensure_shape!`.
- Device memory is owned by `Weights`, `KvCache`, `Workspace` only. Kernel wrappers take views and never allocate.
  The generation loop performs no device allocation.
- Layouts are row-major, contiguous, FP16 unless DESIGN.md §2 says otherwise. Logits are FP32.
- Kernels are validated against a CPU reference test before they are used.
- No ML frameworks (candle, burn, tch, ...) as dependencies.
- Minimal comments: `// SAFETY:` on unsafe blocks and at most a one-line note for non-obvious tricks.
  Longer explanations go in DESIGN.md.
- Tests are deterministic: seeded RNGs (`StdRng::seed_from_u64`), no wall-clock or network dependence.

## Hard rules

- Never loosen a test tolerance or threshold, or delete/ignore a test, to make the gate pass.
  If a tolerance is genuinely wrong, stop and explain in the PR; a human decides.
- Never push to `main`, never force-push, never merge PRs. Work on `agent/<task-id>` branches.
- Never add a dependency without stating why in the PR. Licenses must pass `cargo deny`.
- Overturning a decision recorded in `docs/adr/` requires a new ADR in the same PR.
- When a status changes, update DESIGN.md §7, the backlog entry and `CHANGELOG.md` (Unreleased).
- Commits are signed off (`git commit -s`), one logical change per commit.

## Work selection (deterministic)

The backlog is `.agent/backlog.yaml`. A task is **eligible** if it is `status: todo` and its `depends_on` entries are all `done`.

1. If `make ci-local` fails on `main`, the only task is `fix-<YYYY-MM-DD>`: make main green, nothing else.
2. Single run (`/maintain`): the first eligible task in file order.
3. Parallel run (`/maintain parallel`): an orchestrator takes every eligible task whose `files` do not overlap with
   another selected task or an open `agent/*` PR, ignoring the shared bookkeeping files (CHANGELOG.md, DESIGN.md §7,
   `.agent/backlog.yaml`, `[[test]]` blocks in `engine/Cargo.toml`, new Makefile targets). Each task runs in its own
   subagent and `git worktree` on `agent/<task-id>`, one task per branch and per PR.
4. If nothing is eligible, run triage (see `/maintain triage`) and stop.

Conflicts with `main` are resolved by `git merge origin/main` into the agent branch, never by rebase or force-push.

A task is done only when `make ci-local` and every command in its `acceptance` list exit 0 locally and the PR's required
GitHub checks are green.
Each run writes `.agent/runs/<YYYY-MM-DD>-<task-id>.md` and commits it with the change.

In Claude Code, the full procedure is the `/maintain` skill (`.claude/skills/maintain/SKILL.md`).
