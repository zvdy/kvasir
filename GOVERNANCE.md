# Governance

## Roles

- **Maintainers** (listed in [MAINTAINERS.md](MAINTAINERS.md)) review and merge PRs, own the roadmap
  (`.agent/backlog.yaml`, DESIGN.md) and cut releases.
- **Contributors**: anyone sending issues or PRs.
- **Agents**: automated contributors that follow [AGENTS.md](AGENTS.md). They open PRs only; a maintainer merges.

## Decisions

Day-to-day changes are decided by review: one maintainer approval and green CI. Architectural decisions are recorded
as ADRs in `docs/adr/` and need maintainer approval. While there is a single maintainer, that maintainer decides.
With two or more, disagreements are settled by lazy consensus, then simple majority.

## Becoming a maintainer

A contributor with sustained, high-quality contributions may be nominated by a maintainer. Nominations are approved by
a majority of existing maintainers and recorded by a PR to MAINTAINERS.md.

## Changes to this document

By PR, approved by a majority of maintainers.
