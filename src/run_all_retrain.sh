#!/usr/bin/env bash
# Master unattended runner: main new-substrate retrain (all 4 algos, 240 runs) then the
# Pareto objective-weight sweep (PPO, all_sources, 4 weight points). Sequential, because
# the 8 vCPUs have zero idle headroom — running them concurrently would slow both.
set -u
cd ~/optena
source venv/bin/activate
ALGOS="SAC PPO TD3 A2C"
LOG=~/optena/logs_retrain_all.txt

echo "=== MAIN RETRAIN START $(date -u) ===" | tee "$LOG"
nohup python src/train_rl_temporal.py --worker-id 0 --total-workers 2 --algos $ALGOS >> ~/optena/logs_temporal_w0.txt 2>&1 &
P0=$!
nohup python src/train_rl_temporal.py --worker-id 1 --total-workers 2 --algos $ALGOS >> ~/optena/logs_temporal_w1.txt 2>&1 &
P1=$!
echo "main workers PID $P0 $P1" | tee -a "$LOG"
wait $P0 $P1
echo "=== MAIN RETRAIN DONE $(date -u) ===" | tee -a "$LOG"

# --- Pareto sweep (AWS-7): PPO, all_sources, SLA weight fixed at 0.1 ---
run_pareto () {
  echo ">>> pareto $1 (c=$2 carbon=$3 water=$4) $(date -u)" | tee -a "$LOG"
  python src/train_rl_temporal.py --worker-id 0 --total-workers 1 --algos PPO \
    --configs all_sources --tag "$1" \
    --alpha-cost "$2" --alpha-carbon "$3" --alpha-water "$4" --alpha-sla 0.1 \
    >> ~/optena/logs_pareto.txt 2>&1
}
echo "=== PARETO SWEEP START $(date -u) ===" | tee -a "$LOG"
run_pareto pareto_cost   0.7 0.1 0.1
run_pareto pareto_carbon 0.1 0.7 0.1
run_pareto pareto_water  0.1 0.1 0.7
run_pareto pareto_equal  0.3 0.3 0.3
echo "=== ALL DONE $(date -u) ===" | tee -a "$LOG"
