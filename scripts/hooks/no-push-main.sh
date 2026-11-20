#!/usr/bin/env bash
# PreToolUse hook: refuse git push from main, force pushes, and pushes targeting main.
cmd=$(jq -r '.tool_input.command // ""')
case "$cmd" in *"git push"*) ;; *) exit 0 ;; esac
branch=$(git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$branch" = main ] || echo "$cmd" | grep -Eq -- '(--force|-f( |$)|[: ]main( |$))'; then
  echo "blocked: agents never push to main or force-push; work on an agent/* branch" >&2
  exit 2
fi
exit 0
