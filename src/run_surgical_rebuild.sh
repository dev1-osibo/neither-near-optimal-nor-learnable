#!/usr/bin/env bash
# SURGICAL rebuild after the gas-EF fix (0.41). Retrains ONLY what depends on the gas
# factor; the 7 non-gas main-verdict configs are proven EF-independent and kept.
# PRECONDITION: _drop_gas_from_progress.py has already been run + verified (main progress
# now lists only the 140 non-gas runs as completed). Server shared with medical project:
# never shut down.
set -u
cd ~/optena
LOG=~/optena/surgical_rebuild_run.log
echo "=== SURGICAL REBUILD START (gas EF = 0.41) $(date -u) ===" >> "$LOG"

# --- 1) MAIN VERDICT gas configs (resume: 140 non-gas skipped, 100 gas retrained) ---
echo "[main-gas] start $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 2 \
    --algos SAC PPO TD3 A2C >> "$LOG" 2>&1 &
P0=$!
venv/bin/python src/train_rl_temporal.py --worker-id 1 --total-workers 2 \
    --algos SAC PPO TD3 A2C >> "$LOG" 2>&1 &
P1=$!
wait $P0 $P1
echo "[main-gas] done $(date -u)" >> "$LOG"

# --- 2) PARETO sweep (all_sources is a gas config -> full re-run). Wipe tainted first. ---
echo "[pareto] start $(date -u)" >> "$LOG"
rm -rf models_temporal_pareto_* checkpoints_temporal_pareto_*
rm -f results/rl_results_temporal_pareto_*_worker_*.json
for wp in "cost 0.7 0.1 0.1" "carbon 0.1 0.7 0.1" "water 0.1 0.1 0.7" "equal 0.3 0.3 0.3"; do
  set -- $wp; tag=$1; ac=$2; ak=$3; aw=$4
  venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 1 \
    --algos PPO --configs all_sources --tag pareto_${tag} \
    --alpha-cost ${ac} --alpha-carbon ${ak} --alpha-water ${aw} --alpha-sla 0.1 >> "$LOG" 2>&1
done
echo "[pareto] done $(date -u)" >> "$LOG"

# --- 3) ORACLE arm: FULL clean re-run (wipe the 10 old-EF runs) for uniform provenance ---
echo "[oracle] start $(date -u)" >> "$LOG"
rm -rf models_temporal_oracle checkpoints_temporal_oracle
rm -f results/rl_results_temporal_oracle_worker_*.json
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
Q0=$!
venv/bin/python src/train_rl_temporal.py --worker-id 1 --total-workers 2 \
    --algos PPO --forecast-mode oracle --tag oracle >> "$LOG" 2>&1 &
Q1=$!
wait $Q0 $Q1
echo "[oracle] done $(date -u)" >> "$LOG"

# --- 4) size40 sensitivity (all_sources, battery 40MWh) ---
echo "[size40] start $(date -u)" >> "$LOG"
venv/bin/python src/train_rl_temporal.py --worker-id 0 --total-workers 1 \
    --algos PPO --configs all_sources --tag size40 \
    --battery-capacity-kwh 40000 --battery-max-rate-kw 20000 >> "$LOG" 2>&1
echo "[size40] done $(date -u)" >> "$LOG"
echo "=== ALL DONE $(date -u) ===" >> "$LOG"
