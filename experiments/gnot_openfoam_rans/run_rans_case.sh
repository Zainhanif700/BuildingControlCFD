#!/usr/bin/env bash
# Mesh and run one OpenFOAM case (conda env "foam" active), on one core.
# Usage: bash run_rans_case.sh cases/<name>_dx0.1
set -eu
CASE="${1:?usage: bash run_rans_case.sh <case dir>}"
cd "$CASE"
command -v pimpleFoam > /dev/null || { echo "OpenFOAM not found -- run: conda activate foam"; exit 1; }
say() { echo "[$(date '+%T')] $*"; }
trap 'echo "FAILED at: $BASH_COMMAND -- see the newest log.* file in $CASE"; ls -t log.* 2>/dev/null | head -1' ERR
rm -rf 0 processor* constant/polyMesh
find . -maxdepth 1 -regextype posix-extended -regex './[0-9.]+(e[-+]?[0-9]+)?' ! -name '0.orig' -exec rm -rf {} +
say "mesh: blockMesh";                 blockMesh > log.blockMesh 2>&1
say "mesh: remove the columns";        cp system/topoSetDict.fluid system/topoSetDict
                                       topoSet > log.topoSet.fluid 2>&1
                                       subsetMesh fluid -patch columns -overwrite > log.subsetMesh 2>&1
say "mesh: window/door patches";       cp system/topoSetDict.openings system/topoSetDict
                                       topoSet > log.topoSet.openings 2>&1
                                       createPatch -overwrite > log.createPatch 2>&1
say "mesh: seating zone";              cp system/topoSetDict.seats system/topoSetDict
                                       topoSet > log.topoSet.seats 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Mesh OK|Failed|\*\*\*" log.checkMesh | head -8
grep -E "seats.*size|cellZone" log.topoSet.seats | tail -2 || true
rm -rf 0 && cp -r 0.orig 0
postProcess -func writeCellCentres -time 0 > log.cellCentres 2>&1
APP=$(foamDictionary -entry application -value system/controlDict)
say "run: $APP (k-omega SST) on 1 core (log.$APP)"
$APP > log.$APP 2>&1
if [ "$APP" = pimpleFoam ]; then         # transient: keep p/phi only at the comparison times
    for d in [0-9]*; do
        case "$d" in 0|0.orig|30|60|120|180) ;; *) rm -f "$d"/p "$d"/p.gz "$d"/phi "$d"/phi.gz ;; esac
    done
else                                      # steady: report convergence
    grep -E "SIMPLE solution converged|End" log.$APP | tail -2 || true
    grep -E "Solving for (Ux|p|k|omega)," log.$APP | tail -4 | sed 's/, Final.*//' || true
fi
say "disk use of this case: $(du -sh . | cut -f1)"
say "done. last time step:"
grep -E "^Time =" log.$APP | tail -1
grep -E "Courant Number" log.$APP | tail -1 || true
grep -A3 -E "^yPlus" log.$APP | tail -4 || true
