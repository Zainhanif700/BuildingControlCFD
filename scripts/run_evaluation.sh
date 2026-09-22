#!/usr/bin/env bash
# One-command reproduction of "our" ensemble test-error result (same metric
# as the paper's Table 3), using our own retrained 5-model ensemble in
# learning/data/checkpoints/ensemble_5/. Prints each model's error plus the
# ensemble average, next to the paper's reported ensemble error (10.90%).
#
# To evaluate the paper authors' own checkpoints instead:
#   ./run_evaluation.sh ../local/models/*.pt
set -e
cd "$(dirname "$0")/learning"
if [ "$#" -eq 0 ]; then
    python evaluate_ensemble.py data/checkpoints/ensemble_5/*.pt
else
    python evaluate_ensemble.py "$@"
fi
