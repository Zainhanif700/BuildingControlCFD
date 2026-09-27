#!/usr/bin/env bash
# Mesh + run one OpenFOAM case made by make_openfoam_case.py.
# Usage (conda env "foam" active):  bash run_openfoam_case.sh cases/W1_1ms_dx0.1 [n_procs]
#   --mesh-only as 3rd argument: stop after the mesh (to inspect checkMesh first)
set -eu
CASE="${1:?usage: bash run_openfoam_case.sh <case dir> [n_procs] [--mesh-only]}"
NP="${2:-4}"
cd "$CASE"
command -v pimpleFoam > /dev/null || { echo "OpenFOAM not found -- run: conda activate foam"; exit 1; }
say() { echo "[$(date '+%T')] $*"; }

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

say "run: pimpleFoam on $NP cores (log.pimpleFoam)"
decomposePar -force > log.decomposePar 2>&1
mpirun -np "$NP" pimpleFoam -parallel > log.pimpleFoam 2>&1
reconstructPar > log.reconstructPar 2>&1
rm -rf processor*
say "done. last time step:"
grep -E "^Time =" log.pimpleFoam | tail -1
grep -E "Courant Number" log.pimpleFoam | tail -1
