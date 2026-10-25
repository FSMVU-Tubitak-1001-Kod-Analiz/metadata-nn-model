#!/usr/bin/env bash
# Fusion methods (configs/grid_fusions.yaml, frozen CodeBERT) and token-level cross-attention between
# the patch tokens and the change-text tokens (finetune --coattn, CodeBERT frozen). Waits until
# finetune_metatext.sh has finished, so it can be started right away.
#   mkdir -p logs && PYTHONPATH=src nohup bash run_fusions.sh > logs/run_fusions.log 2>&1 &
set -euo pipefail
C=configs/balanced.yaml
RUNS=artifacts/balanced/runs

while pgrep -f finetune_metatext.sh > /dev/null; do
    echo "waiting for finetune_metatext.sh  $(date '+%F %T')"
    sleep 600
done

echo "=== fusion grid  $(date '+%F %T')"
python -m commitclf -c $C grid --grid configs/grid_fusions.yaml

NAME=finetune/fusion/coattn_repo_in-patch_ctxts_frozen
if [ -f "$RUNS/$NAME/summary.csv" ]; then
    echo "skip $NAME (done)"
else
    echo "=== cross-attention  $(date '+%F %T')"
    python -m commitclf -c $C finetune --model code --code patch --split repo --folds all --code-ft \
        --meta-text change_short --coattn --freeze-code --name "$NAME"
fi
echo "=== all done  $(date '+%F %T')"
echo "Results saved in $RUNS/fusion, $RUNS/finetune/fusion"
