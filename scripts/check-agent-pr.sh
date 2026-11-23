#!/usr/bin/env bash
# Verifies an agent/* PR carries its run log (with local GPU gate evidence), CHANGELOG entry and backlog status.
# Usage: scripts/check-agent-pr.sh [base-ref] [head-ref] [branch-name]
set -euo pipefail
cd "$(dirname "$0")/.."

base=${1:-origin/main}
head=${2:-HEAD}
branch=${3:-$(git rev-parse --abbrev-ref HEAD)}

case "$branch" in
  agent/*) task=${branch#agent/} ;;
  *) echo "not an agent branch ($branch): nothing to check"; exit 0 ;;
esac

fail=0
err() { echo "::error::$1"; fail=1; }

changed=$(git diff --name-only "$base...$head")

log=$(printf '%s\n' "$changed" | grep -E "^\.agent/runs/[0-9]{4}-[0-9]{2}-[0-9]{2}-${task}\.md$" | head -1 || true)
if [ -z "$log" ]; then
  err "missing run log .agent/runs/<YYYY-MM-DD>-${task}.md"
else
  grep -q '^## gate' "$log" || err "$log has no '## gate' section"
  grep -Eq 'test result: ok\.' "$log" || err "$log has no 'test result: ok.' line from the local make test-gpu run"
  ! grep -Eq 'test result: FAILED' "$log" || err "$log records a failing test run"
fi

printf '%s\n' "$changed" | grep -qx 'CHANGELOG.md' || err "CHANGELOG.md not updated"

if grep -Eq "^  - id: ${task}\$" .agent/backlog.yaml; then
  printf '%s\n' "$changed" | grep -qx '.agent/backlog.yaml' || err ".agent/backlog.yaml not updated for ${task}"
  status=$(awk -v id="$task" '$0 ~ "^  - id: "id"$" {f=1; next} f && /^  - id:/ {exit} f && /status:/ {print $2; exit}' .agent/backlog.yaml)
  [ "$status" != todo ] || err "${task} is still status: todo in .agent/backlog.yaml"
fi

[ "$fail" = 0 ] && echo "agent PR checks passed for ${task}"
exit "$fail"
