#!/usr/bin/env bash
# Continuation driver for the epoch-50 line: wait for the full scoring to
# finish, then chain audit -> distillation -> training, each writing to its own
# NEW directory (no epoch-9 files are touched).
#
# Usage: bash FLARE-VLA/commands/libero_plus/15_continue_epoch50_pipeline.sh

set -euo pipefail

MASTER_LOG=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/epoch50_pipeline_master.log
SCORING_SUMMARY=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/teacher_scoring_epoch50/scoring_run_summary.json
PY=/root/autodl-tmp/envs/LeWorldModel/bin/python

log() { echo "[$(date '+%F %T')] $*" >> "$MASTER_LOG"; }

log "=== epoch-50 continuation driver started ==="

# --- 1. wait for full scoring ---
log "waiting for full scoring to finish..."
MAX_WAIT=$((8 * 3600))
elapsed=0
while [ ! -f "$SCORING_SUMMARY" ]; do
    sleep 60
    elapsed=$((elapsed + 60))
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        log "TIMEOUT waiting for scoring (${MAX_WAIT}s)"; exit 1
    fi
done
log "scoring summary found after ~${elapsed}s"

STATUS=$("$PY" -c "import json;print(json.load(open('$SCORING_SUMMARY'))['status'])")
log "scoring status: $STATUS"
if [ "$STATUS" != "PASS" ]; then
    log "scoring did not PASS; aborting before audit"; exit 1
fi

# --- 2. audit ---
log "=== step 2: teacher quality audit ==="
bash /root/autodl-tmp/FLARE-VLA/commands/libero_plus/12_audit_epoch50.sh >> "$MASTER_LOG" 2>&1

# --- 3. distillation dataset ---
log "=== step 3: build distillation dataset ==="
bash /root/autodl-tmp/FLARE-VLA/commands/libero_plus/13_build_distillation_epoch50.sh >> "$MASTER_LOG" 2>&1

# --- 4. training ---
log "=== step 4: GR00T FLARE training (epoch-50 teachers) ==="
bash /root/autodl-tmp/FLARE-VLA/commands/libero_plus/14_train_gr00t_flare_epoch50.sh >> "$MASTER_LOG" 2>&1

log "=== epoch-50 pipeline complete ==="
