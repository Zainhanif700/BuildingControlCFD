#!/usr/bin/env bash
# Run OpenFOAM for all settings of a scenario file (CPU only). Finished cases are kept in cases/ and skipped next time.
# Usage: bash run_batch.sh <scenario file> [cases at a time] [cores per case] [names ...]
#   e.g. bash run_batch.sh scenarios_v2.txt 8 4      (8 cases at a time, 4 cores each)
#   DX=0.08 mesh size (default), T_END=20 short test run
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCEN="${1:?usage: bash run_batch.sh <scenario file> [cases at a time] [cores per case] [names ...]}"; shift
NPAR="${1:-1}"; [ $# -gt 0 ] && shift
export FOAM_NP="${1:-1}"; [ $# -gt 0 ] && shift
export SKIP_EXTRACT=1
DX="${DX:-0.08}"
[ -f "$SCEN" ] || SCEN="$HERE/$SCEN"
[ -f "$SCEN" ] || { echo "scenario file not found: $SCEN"; exit 1; }
cd "$HERE" || exit 1
mkdir -p logs cases
echo "[$(date '+%F %T')] $SCEN${*:+ (only $*)}: $NPAR at a time, $FOAM_NP core(s) each, dx $DX"
# line: name split V1,...,V8 [door1,door2] ...  -> without a doors column both doors are open
grep -v '^#' "$SCEN" | awk -v only="$*" -v dx="$DX" '
    BEGIN { n = split(only, a, " "); for (i = 1; i <= n; i++) w[a[i]] = 1 }
    NF && (n == 0 || ($1 in w)) { print $1, $3, ($4 ~ /^[01],[01]$/ ? $4 : "1,1"), dx }' \
    | xargs -P "$NPAR" -n 4 bash "$HERE/run_one_rans.sh"
echo "[$(date '+%F %T')] finished cases: $(ls -d cases/*_dx${DX}/done 2>/dev/null | wc -l)"
