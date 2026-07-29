#!/usr/bin/env bash
# Chained unattended run for Paper 1 Gates 3 + 4.
#  Gate 3 (oracle arm): PPO x 12 configs x 5 seeds under forecast_mode=oracle
#         (perfect-foresight ablation). Tagged 'oracle' -> isolated from the main
#         verdict glob (rl_results_temporal_worker_*.json). 2 workers in parallel.
#  Gate 4 (sizing): PPO x all_sources x 5 seeds at battery 40MWh/20MW (persistence),
#         tagged 'size40' -> tests whether RL's storage-degradation is size-robust.
# Hyperparameters/timesteps identical to the main run so the ONLY difference is the
# ablated variable. Server is shared with a medical project: do NOT shut it down.
set -u
cd ~/optena
LOG=~/optena/oracle_sizing_run.log
echo "=== START $(date -u) ===" >> "$LOG"

# --- Gate 3: oracle-foresight PPO arm, 2 parallel workers (60 runs total) ---
echo "[oracle] launching 2 workers $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
P0=$!
venv/bin/python src/train_rl_temporal.py --worker-id 1 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
P1=$!
wait $P0 $P1
echo "[oracle] both workers done $(date -u)" >> "$LOG"

# --- Gate 4: sizing sensitivity (persistence, larger battery) ---
echo "[size40] launching $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 1 \
    --algos PPO --configs all_sources --tag size40 \
    --battery-capacity-kwh 40000 --battery-max-rate-kw 20000 >> "$LOG" 2>&1
echo "[size40] done $(date -u)" >> "$LOG"
echo "=== ALL DONE $(date -u) ===" >> "$LOG"
