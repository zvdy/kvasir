#!/usr/bin/env bash
# PreToolUse hook: refuse git push from main, force pushes, and pushes targeting main.
cmd=$(jq -r '.tool_input.command // ""')
pushes=$(printf '%s\n' "$cmd" | sed 's/&&\|||\|;\||/\n/g' | grep -E '^\s*git\s+push(\s|$)')
[ -z "$pushes" ] && exit 0
branch=$(git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$branch" = main ] || printf '%s\n' "$pushes" | grep -Eq -- '(--force|\s-f(\s|$)|[:[:space:]+]main(\s|$))'; then
  echo "blocked: agents never push to main or force-push; work on an agent/* branch" >&2
  exit 2
fi
exit 0
