## Task

<!-- backlog id (e.g. S1-03), fix-YYYY-MM-DD, triage-YYYY-MM-DD, or "none" for human PRs -->

## Summary

## Gate

<!-- last lines of `make ci-local` and each acceptance command with its exit code -->

```
```

## GPU tests

<!-- GPU model and driver from `make doctor`; GitHub CI cannot run these -->

## Checklist

- [ ] `make ci-local` passes on a GPU machine
- [ ] No test tolerance loosened, no test removed or ignored
- [ ] DESIGN.md, `.agent/backlog.yaml` and CHANGELOG.md updated where status changed
- [ ] New ADR if a decision in `docs/adr/` is overturned
- [ ] Commits signed off (`git commit -s`)

## Risks / follow-ups
