## Task

<!-- backlog id (e.g. S1-03), fix-YYYY-MM-DD, triage-YYYY-MM-DD, or "none" for human PRs -->

## Summary

## Gate

<!-- last lines of `make ci-local` and each acceptance command with its exit code -->

```
```

## GPU tests

<!-- GitHub CI has no GPU (ADR 0006). GPU model/driver from `make doctor`, then the `test result:` lines of `make test-gpu`.
Agent PRs must also have them in the run log; the `agent-pr` check enforces it. -->

```
```

## Checklist

- [ ] `make ci-local` passes on a GPU machine
- [ ] Required GitHub checks green: `ci-pass`, `dco`, `agent-pr`
- [ ] No test tolerance loosened, no test removed or ignored
- [ ] DESIGN.md, `.agent/backlog.yaml` and CHANGELOG.md updated where status changed
- [ ] New ADR if a decision in `docs/adr/` is overturned
- [ ] Commits signed off (`git commit -s`)

## Risks / follow-ups
