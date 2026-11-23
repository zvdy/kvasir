#!/usr/bin/env bash
# PreToolUse hook: refuse git push from main, force pushes, and pushes targeting main.
# The branch is taken from the directory the push runs in (`git -C <dir>` or the tool call's cwd), so worktrees work.
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
cwd=${cwd:-${CLAUDE_PROJECT_DIR:-.}}
pushes=$(printf '%s\n' "$cmd" | sed 's/&&\|||\|;\||/\n/g' | grep -E '^\s*git(\s+-C\s+\S+)?\s+push(\s|$)')
[ -z "$pushes" ] && exit 0
blocked=0
while IFS= read -r p; do
  dir=$(printf '%s\n' "$p" | sed -nE 's/^\s*git\s+-C\s+(\S+)\s+push.*/\1/p')
  dir=${dir:-$cwd}
  case "$dir" in /*) ;; *) dir="$cwd/$dir" ;; esac
  branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null)
  if [ -z "$branch" ] || [ "$branch" = main ] ||
    printf '%s\n' "$p" | grep -Eq -- '(--force|\s-f(\s|$)|[:[:space:]+]main(\s|$))'; then
    blocked=1
  fi
done <<< "$pushes"
if [ "$blocked" = 1 ]; then
  echo "blocked: agents never push to main or force-push; work on an agent/* branch" >&2
  exit 2
fi
exit 0
