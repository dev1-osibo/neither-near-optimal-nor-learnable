#!/usr/bin/env bash
# FULL CLEAN REBUILD after the gas-carbon fix (0.00041 -> 0.41 kgCO2/kWh).
# Single-substrate provenance: wipes ALL prior (gas-EF-tainted) training outputs and
# re-runs the main verdict, Pareto sweep, oracle-foresight arm, and size40 sensitivity
# on the corrected substrate. Server is SHARED with a medical project -> never shut down.
set -u
cd ~/optena
LOG=~/optena/full_rebuild_run.log
echo "=== FULL REBUILD START (gas EF = 0.41) $(date -u) ===" >> "$LOG"

# --- Clean all prior training artifacts (tainted); local backups retain the old copies ---
rm -rf models_temporal models_temporal_oracle models_temporal_pareto_* models_temporal_size40 \
       checkpoints_temporal checkpoints_temporal_oracle checkpoints_temporal_pareto_* \
       checkpoints_temporal_size40
rm -f results/rl_results_temporal_worker_*.json \
      results/rl_results_temporal_oracle_worker_*.json \
      results/rl_results_temporal_pareto_*_worker_*.json \
      results/rl_results_temporal_size40_worker_*.json
echo "CLEANED $(date -u)" >> "$LOG"

# --- 1) MAIN VERDICT: 4 algos x 12 configs x 5 seeds, 2 workers, persistence forecast ---
echo "[main] start $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 2 \
    --algos SAC PPO TD3 A2C >> "$LOG" 2>&1 &
P0=$!
venv/bin/python src/train_rl_temporal.py --worker-id 1 --total-workers 2 \
    --algos SAC PPO TD3 A2C >> "$LOG" 2>&1 &
P1=$!
wait $P0 $P1
echo "[main] done $(date -u)" >> "$LOG"

# --- 2) PARETO sweep: PPO x all_sources x 4 weight points (SLA fixed 0.1) ---
echo "[pareto] start $(date -u)" >> "$LOG"
for wp in "cost 0.7 0.1 0.1" "carbon 0.1 0.7 0.1" "water 0.1 0.1 0.7" "equal 0.3 0.3 0.3"; do
  set -- $wp; tag=$1; ac=$2; ak=$3; aw=$4
  venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 1 \
    --algos PPO --configs all_sources --tag pareto_${tag} \
    --alpha-cost ${ac} --alpha-carbon ${ak} --alpha-water ${aw} --alpha-sla 0.1 >> "$LOG" 2>&1
done
echo "[pareto] done $(date -u)" >> "$LOG"

# --- 3) ORACLE-foresight arm: PPO x 12 configs x 5 seeds, 2 workers, forecast=oracle ---
echo "[oracle] start $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
Q0=$!
venv/bin/python src/train_rl_temporal.py --worker-id 1 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
Q1=$!
wait $Q0 $Q1
echo "[oracle] done $(date -u)" >> "$LOG"

# --- 4) size40 sensitivity: PPO x all_sources x 5 seeds, battery 40MWh/20MW, persistence ---
echo "[size40] start $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 1 \
    --algos PPO --configs all_sources --tag size40 \
    --battery-capacity-kwh 40000 --battery-max-rate-kw 20000 >> "$LOG" 2>&1
echo "[size40] done $(date -u)" >> "$LOG"
echo "=== ALL DONE $(date -u) ===" >> "$LOG"
