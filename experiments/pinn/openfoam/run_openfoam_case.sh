#!/usr/bin/env bash
# Mesh and run one laminar OpenFOAM reference case (conda env "foam" active).
# Usage: bash run_openfoam_case.sh cases/W1_1ms_dx0.1 [n_procs] [--mesh-only]
set -eu
CASE="${1:?usage: bash run_openfoam_case.sh <case dir> [n_procs] [--mesh-only]}"
NP="${2:-4}"
cd "$CASE"
command -v pimpleFoam > /dev/null || { echo "OpenFOAM not found -- run: conda activate foam"; exit 1; }
say() { echo "[$(date '+%T')] $*"; }
trap 'echo "FAILED at: $BASH_COMMAND -- see the newest log.* file in $CASE"; ls -t log.* 2>/dev/null | head -1' ERR

# clean leftovers of an earlier run: an existing 0/ (fields with window/door patches that do not
# exist yet on the fresh blockMesh mesh) makes subsetMesh fail; also old sets, time dirs, processors
rm -rf 0 processor* constant/polyMesh
find . -maxdepth 1 -regextype posix-extended -regex './[0-9.]+(e[-+]?[0-9]+)?' ! -name '0.orig' -exec rm -rf {} +

say "mesh: blockMesh";                 blockMesh > log.blockMesh 2>&1
say "mesh: remove the columns";        cp system/topoSetDict.fluid system/topoSetDict
                                       topoSet > log.topoSet.fluid 2>&1
                                       subsetMesh fluid -patch columns -overwrite > log.subsetMesh 2>&1
say "mesh: window/door patches";       cp system/topoSetDict.openings system/topoSetDict
                                       topoSet > log.topoSet.openings 2>&1
                                       createPatch -overwrite > log.createPatch 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Mesh OK|Failed|\*\*\*" log.checkMesh | head -8
echo "patches (name, faces):"
awk '/^Checking patch topology/{f=1} f&&/ok|multiply/{print "  "$1, $2}' log.checkMesh | head -20
rm -rf 0 && cp -r 0.orig 0
postProcess -func writeCellCentres -time 0 > log.cellCentres 2>&1
[ "${3:-}" = "--mesh-only" ] && { say "mesh done (--mesh-only)"; exit 0; }

if [ "$NP" -le 1 ]; then
    say "run: pimpleFoam on 1 core (log.pimpleFoam)"
    pimpleFoam > log.pimpleFoam 2>&1
else
    say "run: pimpleFoam on $NP cores (log.pimpleFoam)"
    decomposePar -force > log.decomposePar 2>&1
    mpirun -np "$NP" pimpleFoam -parallel > log.pimpleFoam 2>&1
    reconstructPar > log.reconstructPar 2>&1
    rm -rf processor*
fi
# disk: keep the velocity at every saved time (the CO2 transport in compare_with_openfoam.py needs
# them) but pressure and fluxes only at the comparison times 30/60/120 s (the server disk filled up)
for d in [0-9]*; do
    case "$d" in 0|0.orig|30|60|120) ;; *) rm -f "$d"/p "$d"/p.gz "$d"/phi "$d"/phi.gz ;; esac
done
say "disk use of this case: $(du -sh . | cut -f1)"
say "done. last time step:"
grep -E "^Time =" log.pimpleFoam | tail -1
grep -E "Courant Number" log.pimpleFoam | tail -1
