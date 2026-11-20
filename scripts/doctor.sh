#!/usr/bin/env bash
# Prints one key=value per check. Exit 1 if a required check fails.
set -u
cd "$(dirname "$0")/.."

fail=0
req() { if [ "$2" = ok ]; then echo "$1=ok${3:+ ($3)}"; else echo "$1=FAIL${3:+ ($3)}"; fail=1; fi; }
opt() { if [ "$2" = ok ]; then echo "$1=ok${3:+ ($3)}"; else echo "$1=no${3:+ ($3)}"; fi; }

want_rust=$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)
have_rust=$(rustc --version 2>/dev/null | awk '{print $2}')
[ "$have_rust" = "$want_rust" ] && r=ok || r=bad
req rustc "$r" "want $want_rust, have ${have_rust:-none}"

nvcc=${NVCC:-}
if [ -z "$nvcc" ]; then
  for root in "${CUDA_PATH:-}" "${CUDA_HOME:-}" /opt/cuda /usr/local/cuda; do
    [ -n "$root" ] && [ -x "$root/bin/nvcc" ] && { nvcc="$root/bin/nvcc"; break; }
  done
fi
[ -z "$nvcc" ] && nvcc=$(command -v nvcc || true)
ver=$([ -n "$nvcc" ] && "$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ -n "$ver" ] && r=ok || r=bad
req nvcc "$r" "${ver:-not found}"
[ "${ver%%.*}" = 13 ] && r=ok || r=bad
req cuda_major_13 "$r"

gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
[ -n "$gpu" ] && r=ok || r=bad
req gpu "$r" "${gpu:-none}"

command -v cargo-deny >/dev/null && r=ok || r=bad
req cargo_deny "$r"

gh auth status >/dev/null 2>&1 && r=ok || r=bad
opt gh_auth "$r"

[ -f models/Llama-3.2-1B/model.safetensors ] && r=ok || r=bad
opt weights "$r" "models/Llama-3.2-1B"

[ -z "$(git status --porcelain)" ] && r=ok || r=bad
opt clean_tree "$r"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = main ] && r=ok || r=bad
opt on_main "$r" "$branch"

git fetch -q origin main 2>/dev/null
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
[ "$behind" = 0 ] && r=ok || r=bad
opt up_to_date "$r" "behind=$behind"

exit $fail
