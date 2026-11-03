# Makes the OpenFOAM commands available (sourced by the other scripts). Order:
#   1. already there (OpenFOAM's etc/bashrc was sourced before, or a Docker/app shell)
#   2. FOAM_BASHRC=/path/to/OpenFOAM/etc/bashrc  -> sourced here
#   3. a conda environment "foam" in ~/anaconda3 (our lab PC)
if ! command -v pimpleFoam > /dev/null 2>&1; then
    _foam_opts="$-"
    set +eu                              # OpenFOAM's bashrc and conda are not written for 'set -eu'
    if [ -n "${FOAM_BASHRC:-}" ]; then
        source "$FOAM_BASHRC"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam
        # the conda package looks for the MPI library in lib/sys-mpich but installs it as lib/mpich-<version>
        L="${CONDA_PREFIX:-}/lib"
        if [ -n "${CONDA_PREFIX:-}" ] && [ ! -e "$L/sys-mpich" ] && ls -d "$L"/mpich-* > /dev/null 2>&1; then
            ln -s "$(basename "$(ls -d "$L"/mpich-* | head -1)")" "$L/sys-mpich"
        fi
    fi
    case "$_foam_opts" in *e*) set -e ;; esac
    case "$_foam_opts" in *u*) set -u ;; esac
fi
command -v pimpleFoam > /dev/null 2>&1 || { echo "OpenFOAM not found: source its etc/bashrc first or set FOAM_BASHRC"; exit 1; }
# the cases are written for OpenFOAM from openfoam.com (versions like v2412); openfoam.org (versions like 12, 14) differs
case "${WM_PROJECT_VERSION:-}" in
    v*|"") ;;
    *) echo "found OpenFOAM $WM_PROJECT_VERSION from openfoam.org -- this project needs OpenFOAM v2412 from openfoam.com"; exit 1 ;;
esac
