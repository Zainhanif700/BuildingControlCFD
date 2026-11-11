#!/usr/bin/env bash
# Continues a finished OpenFOAM case from T0 (needs p and phi saved at T0) to T1, with the save interval of the
# case or a new one. The times after T0 are recomputed and overwritten.
# Usage: bash continue_case.sh <case> [T0, default 120] [T1, default 450] [cores, default 4] [save interval s]
set -euo pipefail
CASE="$(cd "${1:?usage: bash continue_case.sh <case> [T0] [T1] [cores]}" && pwd)"
T0="${2:-120}"; T1="${3:-450}"; NP="${4:-4}"; WI="${5:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../gnot_openfoam_rans/foam_env.sh"
cd "$CASE"
for f in U p phi k omega nut s; do
    [ -f "$T0/$f" ] || [ -f "$T0/$f.gz" ] || { echo "$CASE/$T0/$f missing -- cannot continue from $T0 s"; exit 1; }
done
echo "[$(date '+%F %T')] continue $(basename "$CASE") from $T0 s to $T1 s on $NP cores"
foamDictionary -entry startFrom -set startTime system/controlDict > /dev/null
foamDictionary -entry startTime -set "$T0" system/controlDict > /dev/null
foamDictionary -entry endTime -set "$T1" system/controlDict > /dev/null
[ -z "$WI" ] || foamDictionary -entry writeInterval -set "$WI" system/controlDict > /dev/null
cat > system/decomposeParDict <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains $NP;
method scotch;
EOF
if [ "$NP" -gt 1 ]; then
    decomposePar -force -time "$T0" > log.decomposePar.cont 2>&1
    mpirun -np "$NP" pimpleFoam -parallel > log.pimpleFoam.cont 2>&1
    grep -q "^End" log.pimpleFoam.cont || { echo "OpenFOAM did not reach the end -- see $CASE/log.pimpleFoam.cont"; exit 1; }
    reconstructPar > log.reconstructPar.cont 2>&1
    rm -rf processor*
else
    pimpleFoam > log.pimpleFoam.cont 2>&1
    grep -q "^End" log.pimpleFoam.cont || { echo "OpenFOAM did not reach the end -- see $CASE/log.pimpleFoam.cont"; exit 1; }
fi
for d in [0-9]*; do
    case "$d" in 0|0.orig|"$T0"|"$T1") ;; *) rm -f "$d"/p "$d"/p.gz "$d"/phi "$d"/phi.gz ;; esac
done
echo "[$(date '+%F %T')] done: last time $(foamListTimes -latestTime 2>/dev/null | tail -1), case size $(du -sh . | cut -f1)"
