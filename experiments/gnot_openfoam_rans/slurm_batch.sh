#!/bin/bash
# SLURM version of run_batch.sh: one array task per setting. Submit from this folder:
#   mkdir -p logs && sbatch --array=1-$(grep -v '^#' scenarios_v2.txt | awk 'NF' | wc -l)%8 slurm_batch.sh scenarios_v2.txt
#SBATCH --job-name=classroom-cfd
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%A_%a.out
set -u
SCEN="${1:-scenarios_v2.txt}"
NP="${SLURM_CPUS_PER_TASK:-1}"
NAME=$(grep -v '^#' "$SCEN" | awk 'NF' | sed -n "${SLURM_ARRAY_TASK_ID:?submit with --array}p" | awk '{print $1}')
[ -n "$NAME" ] || { echo "no setting number $SLURM_ARRAY_TASK_ID in $SCEN"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $NAME on $(hostname), $NP cores"
bash run_batch.sh "$SCEN" 1 "$NP" "$NAME"
[ "${T_END:-600}" = 600 ] || NAME="${NAME}_test"
[ -f "cases/${NAME}_dx${DX:-0.08}/done" ] || { echo "$NAME did not finish -- see logs/cases/$NAME.log"; exit 1; }
