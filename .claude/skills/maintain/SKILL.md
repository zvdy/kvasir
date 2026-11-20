---
name: maintain
description: Periodic maintenance run for kvasir. Checks the environment, keeps main green, picks exactly one backlog task, implements it on an agent/* branch and opens a PR. Use when the user types /maintain or asks for a maintenance run. `/maintain triage` only checks health and proposes work.
---

# /maintain

Follow these steps in order. Do not skip or reorder them. AGENTS.md hard rules apply throughout.
Use `<date>` = today as YYYY-MM-DD.

## 1. Environment

Run `make doctor`. If it exits non-zero, stop and report the FAIL lines with the fix for each. Do not change code.
Record the output for the run log.

## 2. Sync and in-flight check

```
git switch main && git pull --ff-only
gh pr list --state open --search "head:agent/" --json number,title,headRefName
```

If the working tree is dirty, stop and report. If an open `agent/*` PR exists, stop and report its number and title.
Do not stack work on unmerged agent PRs.

If `make doctor` reported `weights=ok` and `ENV-weights` is `blocked` in `.agent/backlog.yaml`, treat flipping it to
`done` as part of this run's task (same PR).

## 3. Baseline gate

Run `make ci-local` on main.
- Red: the task is `fix-<date>`. Fix only what is broken, keep the diff minimal, and skip to step 5.
- Green: continue.

## 4. Select the task

Read `.agent/backlog.yaml`. Pick the **first** task in file order with `status: todo` whose `depends_on` are all `done`.
If none, run **triage** (below) and stop. Set the task's `status: in_progress` locally.

If the argument is `triage`, skip to triage now.

## 5. Implement

```
git switch -c agent/<task-id>
```

- Read DESIGN.md sections and the files listed in the task first. Reuse existing wrappers/utilities.
- Stay inside the task's `files`. If you must touch others, say why in the run log.
- Write tests first where possible, and keep them deterministic.

## 6. Gate

Run `make fmt`, then `make ci-local`, then every `acceptance` command. On failure, fix and rerun, with at most **5** full
gate attempts. If it is still red after 5, set `status: blocked` with a `notes:` line saying why, commit what is useful,
and open the PR as a **draft** (step 9) titled `[blocked] <task-id>: ...`.

## 7. Review

Launch the `reviewer` subagent on `git diff main...HEAD`. Fix every FAIL item and rerun step 6 if code changed.

## 8. Bookkeeping

- `.agent/backlog.yaml`: `status: done` (or `blocked`).
- DESIGN.md §7 status table and `CHANGELOG.md` `## [Unreleased]` entry.
- `.agent/runs/<date>-<task-id>.md` using this template:

```
# <date> <task-id>: <title>
## doctor
<make doctor output>
## baseline
green | red (<failing target>)
## changes
- <bullet per logical change, with file paths>
## gate
<last 15 lines of make ci-local> + acceptance commands and exit codes
## review
<reviewer verdict, items fixed>
## follow-ups
- <proposed backlog items, not added to backlog.yaml>
```

## 9. Commit and PR

```
git add -A
git commit -s -m "<task-id>: <title>" -m "<one-paragraph summary>" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin agent/<task-id>
gh pr create --base main --head agent/<task-id> --title "<task-id>: <title>" --body-file <filled .github/pull_request_template.md>
```

Fill every PR template section. Never merge.

## 10. Report

Print: task id, PR URL, gate result, follow-ups. Nothing else.

## Triage (no feature work)

1. Steps 1–3.
2. `git switch -c agent/triage-<date>`, run `cargo update`, then `make ci-local`. If green and `Cargo.lock` changed, commit
   ("deps: cargo update") and open a PR. If red, revert the lockfile and record the failing crate.
3. Propose up to 3 new backlog items (with id, files, acceptance) in the run log's follow-ups. Do not edit
   backlog.yaml order.
