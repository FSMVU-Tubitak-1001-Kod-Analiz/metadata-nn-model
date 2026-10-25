#!/usr/bin/env bash
# Fine-tuned CodeBERT with the metadata written as text (balanced data, held-out repos, 5 folds).
# Needs `metatext` to have been run (data/processed_balanced/metatext.csv). Settings: `finetune` in
# configs/balanced.yaml. Runs that already have a summary.csv are skipped.
#   mkdir -p logs && PYTHONPATH=src nohup bash finetune_metatext.sh > logs/finetune_metatext.log 2>&1 &
set -euo pipefail
C=configs/balanced.yaml
OUT=artifacts/balanced/runs/finetune/metatext

ft() {   # ft <run name> <finetune options...>
    local name=$1; shift
    if [ -f "$OUT/$name/summary.csv" ]; then echo "skip $name (done)"; return; fi
    echo "=== $name  $(date '+%F %T')"
    python -m commitclf -c $C finetune --split repo --folds all --code-ft --name "finetune/metatext/$name" "$@"
}

# reference: fine-tuned CodeBERT on the patch alone
ft code_repo_in-patch               --model code --code patch --no-message
# joint: metadata text + patch as one input pair
ft code_repo_in-patchtxts           --model code --code patch_meta_short --no-message
# separate per file: CodeBERT(patch) and CodeBERT(text), same encoder
ft code_repo_in-patch+txts          --model code --code patch meta_short --no-message
# fusion: CodeBERT(change text) as the metadata side of gated, no numeric metadata
ft gated_repo_in-patch_ctxts_nofacts --model gated --code patch --meta --no-facts --meta-text change_short
echo "=== all done  $(date '+%F %T')"
echo "Results saved in $OUT"
