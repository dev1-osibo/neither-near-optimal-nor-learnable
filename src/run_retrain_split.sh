#!/usr/bin/env bash
# =============================================================================
# FULL RE-TRAIN on the CORRECTED substrate (audit fixes #1 solar, #2 carbon obs,
# #4 battery RT 0.90, #5 water sign), SPLIT across two boxes.
#
# The 445-run campaign is sharded by (i % TOTAL_WORKERS). We run ONE single-threaded
# worker per vCPU (OMP=1; the work is env-stepping-bound) and split worker-ids across boxes:
#     Existing  8-vCPU box (98.90.192.78):  ./run_retrain_split.sh $(seq 0 7)      # ids 0-7
#     New      48-vCPU box (52.54.108.4) :  ./run_retrain_split.sh $(seq 8 55)     # ids 8-55
# TOTAL_WORKERS=56 (8+48). Every worker carries an equal 1/56 share, so both boxes finish
# together. Result JSONs are per-worker; pull all 56 from both boxes to merge.
#
# FULL retrain (not surgical): the carbon-observation fix changes every run, so
# ALL prior models/progress are wiped first. Old result JSONs are already backed
# up locally in results/rl_backup_20260716/.
#
# Server shared with a live medical project: NEVER shut the box down; this only
# runs training processes.
# =============================================================================
set -u
cd ~/optena || { echo "cd ~/optena failed"; exit 1; }
PY="${PY:-venv/bin/python}"
TOTAL=56
WIDS=("$@")
if [ ${#WIDS[@]} -eq 0 ]; then echo "usage: $0 <worker-id> [worker-id ...]"; exit 1; fi
# One single-threaded worker per vCPU: the training bottleneck is Python env-stepping, so
# 1 thread/worker + 1 worker/vCPU maximizes throughput without oversubscription.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=~/optena/retrain_split_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).log
echo "=== FULL RETRAIN (corrected substrate) workers=${WIDS[*]} of $TOTAL  $(date -u) ===" | tee -a "$LOG"

# --- 0) One-time clean wipe of prior (now-superseded) training artifacts ---
#     Guarded by a sentinel so re-runs (resume) do NOT wipe in-progress work.
if [ ! -f .retrain_wiped ]; then
  echo "[wipe] clearing prior models/checkpoints/progress/results $(date -u)" | tee -a "$LOG"
  rm -rf models_temporal* checkpoints_temporal*
  rm -f results/rl_results_temporal*_worker_*.json
  touch .retrain_wiped
fi

launch () {  # launch the local worker-ids for one arm with a common arg set, then wait
  local desc="$1"; shift
  echo "[$desc] start $(date -u)" | tee -a "$LOG"
  local pids=()
  for w in "${WIDS[@]}"; do
    $PY src/train_rl_temporal.py --worker-id "$w" --total-workers "$TOTAL" "$@" >> "$LOG" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[$desc] done $(date -u)" | tee -a "$LOG"
}

# --- 1) MAIN VERDICT: 4 algos x 12 configs x 5 seeds = 240 (persistence, no tag) ---
launch "main" --algos SAC PPO TD3 A2C

# --- 2) FORESIGHT ABLATION: PPO + SAC x 12 x 5 = 120 (oracle foresight, tagged) [audit #6] ---
launch "oracle" --algos PPO SAC --forecast-mode oracle --tag oracle

# --- 3) PARETO SWEEP: PPO x all_sources x 4 weightings x 20 seeds = 80 [audit #12] ---
for wp in "cost 0.7 0.1 0.1" "carbon 0.1 0.7 0.1" "water 0.1 0.1 0.7" "equal 0.3 0.3 0.3"; do
  set -- $wp; tag=$1; ac=$2; ak=$3; aw=$4
  launch "pareto_${tag}" --algos PPO --configs all_sources --n-seeds 20 \
    --tag "pareto_${tag}" --alpha-cost "$ac" --alpha-carbon "$ak" --alpha-water "$aw" --alpha-sla 0.1
done

# --- 4) SIZE40 sensitivity: PPO x all_sources x 5 seeds, 40 MWh battery ---
launch "size40" --algos PPO --configs all_sources --tag size40 \
  --battery-capacity-kwh 40000 --battery-max-rate-kw 20000

echo "=== ALL ARMS DONE on $(hostname) workers=${WIDS[*]}  $(date -u) ===" | tee -a "$LOG"
