# Makes the OpenFOAM commands available (sourced by the other scripts). Order:
#   1. already there (OpenFOAM's etc/bashrc was sourced before, or a Docker/app shell)
#   2. FOAM_BASHRC=/path/to/OpenFOAM/etc/bashrc  -> sourced here
#   3. a conda environment "foam" in ~/anaconda3 (our lab PC)
if ! command -v pimpleFoam > /dev/null 2>&1; then
    set +u
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
    set -u
fi
command -v pimpleFoam > /dev/null 2>&1 || { echo "OpenFOAM not found: source its etc/bashrc first or set FOAM_BASHRC"; exit 1; }
