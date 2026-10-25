#!/usr/bin/env bash
# Pack the run summaries (summary.csv, metrics.csv, spec.json) under one or more folders for upload.
# The archive name comes from the first folder: artifacts/colare1793/runs/win -> results_colare1793_runs_win_<date>.tar.gz
# Files given as arguments (e.g. a log) are added as they are.
#
#   ./pack_results.sh artifacts/colare1793/runs/win
#   ./pack_results.sh artifacts/balanced/runs/metatext metatext.log
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <runs folder> [more folders or files ...]" >&2
    exit 1
fi

dirs=(); files=()
for p in "$@"; do
    if [ -f "$p" ]; then files+=("$p")
    elif [ -d "$p" ]; then dirs+=("${p%/}")
    else echo "not found: $p" >&2; exit 1
    fi
done
if [ ${#dirs[@]} -eq 0 ]; then
    echo "give at least one runs folder" >&2
    exit 1
fi

name=${dirs[0]#./}
name=${name#artifacts/}
name=${name//\//_}
out="results_${name}_$(date +%F).tar.gz"

list=$(mktemp)
trap 'rm -f "$list"' EXIT
find "${dirs[@]}" \( -name summary.csv -o -name metrics.csv -o -name spec.json \) > "$list"
if [ ${#files[@]} -gt 0 ]; then
    printf '%s\n' "${files[@]}" >> "$list"
fi
n=$(grep -c . "$list" || true)
if [ "$n" -eq 0 ]; then
    echo "nothing to pack under ${dirs[*]}" >&2
    exit 1
fi

tar czf "$out" -T "$list"
echo "$n files, $(find "${dirs[@]}" -name summary.csv | wc -l) runs"
ls -lh "$out"
