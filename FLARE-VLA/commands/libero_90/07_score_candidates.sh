#!/usr/bin/env bash
cd /root/autodl-tmp/le-wm

export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

python scripts/flare/05_score_validation_set.py
