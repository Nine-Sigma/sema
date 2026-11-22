#!/usr/bin/env bash
# Refuse text holding a value of a PARTNER_* key in .env or a line of .sensitive-patterns.
# Reports the key name or pattern line number, never the value.
# Usage: scan-sensitive.sh staged | message <file> | range <git log args...>
set -euo pipefail

root=$(git rev-parse --show-toplevel)
common=$(cd "$(git rev-parse --git-common-dir)" && pwd)/..
patterns=$(mktemp)
text=$(mktemp)
trap 'rm -f "$patterns" "$text"' EXIT

for dir in "$root" "$common"; do
  if [ -f "$dir/.env" ]; then
    grep -E '^PARTNER_[A-Z0-9_]+=' "$dir/.env" \
      | sed -E "s/^([A-Z0-9_]+)=[\"']?([^\"']*)[\"']?\$/\1\t\2/" >> "$patterns" || true
  fi
  if [ -f "$dir/.sensitive-patterns" ]; then
    awk '!/^#/ && NF {print ".sensitive-patterns line " NR "\t" $0}' \
      "$dir/.sensitive-patterns" >> "$patterns"
  fi
done
[ -s "$patterns" ] || exit 0

case "${1:-}" in
  staged) git diff --cached -U0 --no-color | grep '^+' | grep -v '^+++' > "$text" || true ;;
  message) cat "$2" > "$text" ;;
  range) shift; git log -p --no-color --format='%H%n%B' "$@" > "$text" ;;
  *) echo "usage: scan-sensitive.sh staged | message <file> | range <args>" >&2; exit 2 ;;
esac

found=0
while IFS=$'\t' read -r name value; do
  [ -n "$value" ] || continue
  if grep -q -F -- "$value" "$text"; then
    echo "refused: the text holds the value of $name" >&2
    found=1
  fi
done < "$patterns"
exit "$found"
