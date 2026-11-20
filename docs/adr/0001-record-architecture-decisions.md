# 1. Record architecture decisions

Status: accepted

## Context

kvasir is maintained partly by agents that start each run without memory. Decisions they cannot see, they may undo.

## Decision

Architecturally significant decisions are recorded as numbered ADRs in `docs/adr/` (context, decision, consequences).
Overturning one requires a new ADR that supersedes it, in the same PR as the change.

## Consequences

Rationale is discoverable by humans and agents; DESIGN.md stays the description of *what* the system is, ADRs record
*why*.
