#!/usr/bin/env bash
# TEMPORARY unattended runner (delete after use):
#   1. wait until the v19 training has finished (final checkpoint + process gone)
#   2. full final checks for v19 with v19's OWN code (git worktree at the v19 commit, GPU)
#   3. v20 smoke test            -> stop if any stage fails
#   4. v20 300-iteration dry run  -> stop on crash or NaN
#   5. full v20 training
#   6. full final checks for v20 (GPU)
# Everything is logged to auto_v19_v20.log (plus the usual per-step logs).
# Usage (from experiments/gnot, in its own tmux window):  bash run_v19_final_then_v20.sh
set -u
cd "$(dirname "$0")"
GNOT=$(pwd)
REPO=$(cd ../.. && pwd)
V19_COMMIT=bb908fa
V19_CKPT="$GNOT/checkpoints/v19_throughflow/gnot_v19_throughflow_final.pth"
WT="$HOME/gnot_v19"
exec > >(tee -a "$GNOT/auto_v19_v20.log") 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }

# preconditions: this checkout must be v20, before waiting hours
grep -q '^VERSION = "v20_co2window"' train_gnot.py || { stamp "ABORT: train_gnot.py here is not v20 (git pull first)"; exit 1; }
[ -d "$WT" ] || git -C "$REPO" worktree add "$WT" "$V19_COMMIT" || { stamp "ABORT: could not create v19 worktree"; exit 1; }
grep -q '^VERSION = "v19_throughflow"' "$WT/experiments/gnot/train_gnot.py" || { stamp "ABORT: $WT is not the v19 code"; exit 1; }

stamp "1/6 waiting for v19 to finish ($V19_CKPT)"
while [ ! -f "$V19_CKPT" ]; do sleep 300; done
while pgrep -f "python3 train_gnot.py" > /dev/null; do sleep 60; done
stamp "v19 training finished"

stamp "2/6 v19 final checks (v19 code, GPU)"
(cd "$WT/experiments/gnot" && bash post_training_checks.sh "$V19_CKPT" --gpu) || stamp "WARNING: some v19 check scripts failed -- see the log above"
cp "$WT/experiments/gnot/checks_gnot_v19_throughflow_final.log" "$GNOT/" 2>/dev/null
stamp "v19 checks done -> checks_gnot_v19_throughflow_final.log"

stamp "3/6 v20 smoke test"
python3 staged_smoke_test.py > smoke_v20.log 2>&1
grep -E "PASS|FAIL|RESULT" smoke_v20.log
grep -q "RESULT: ALL STAGES PASSED" smoke_v20.log || { stamp "ABORT: v20 smoke test failed (smoke_v20.log)"; exit 1; }

stamp "4/6 v20 dry run (300 iterations)"
DRY="dry$(date +%m%d%H%M)"
python3 train_gnot.py --iters 300 --tag "$DRY" > dry_v20.log 2>&1 || { stamp "ABORT: dry run crashed"; tail -20 dry_v20.log; exit 1; }
if grep -Eq "Total=nan|NS=nan|CO2\(scaled\)=nan|Walls=nan|Windows=nan" dry_v20.log; then
    stamp "ABORT: NaN in the v20 dry run (dry_v20.log)"; exit 1
fi
grep "Iter 00300" dry_v20.log
rm -rf "checkpoints/v20_co2window_$DRY"

stamp "5/6 v20 full training (~6.7 h) -> train_v20_co2window.log"
python3 train_gnot.py > train_v20_co2window.log 2>&1 || { stamp "ABORT: v20 training crashed"; tail -30 train_v20_co2window.log; exit 1; }
stamp "v20 training finished"

stamp "6/6 v20 final checks (GPU)"
bash post_training_checks.sh checkpoints/v20_co2window/gnot_v20_co2window_final.pth --gpu || stamp "WARNING: some v20 check scripts failed -- see the log above"
stamp "ALL DONE. Final logs: checks_gnot_v19_throughflow_final.log and checks_gnot_v20_co2window_final.log"
