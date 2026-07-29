#!/usr/bin/env bash
# Launches the leakage-free temporal-split retrain (Option A: all 4 algos,
# 12 configs, 5 seeds = 240 runs) across 2 workers on the 8 vCPUs.
# Train window = 2020-2023, eval window = held-out 2024-2025.
# NO auto-shutdown: other projects share this instance.
set -u
cd ~/optena
source venv/bin/activate

ALGOS="SAC PPO TD3 A2C"
LOG0=~/optena/logs_temporal_w0.txt
LOG1=~/optena/logs_temporal_w1.txt

echo "=== TEMPORAL RETRAIN START $(date -u) ===" | tee ~/optena/logs_temporal_main.txt

nohup python src/train_rl_temporal.py --worker-id 0 --total-workers 2 --algos $ALGOS > "$LOG0" 2>&1 &
echo "worker0 PID $!" | tee -a ~/optena/logs_temporal_main.txt

nohup python src/train_rl_temporal.py --worker-id 1 --total-workers 2 --algos $ALGOS > "$LOG1" 2>&1 &
echo "worker1 PID $!" | tee -a ~/optena/logs_temporal_main.txt

echo "Both workers launched. Monitor: tail -f $LOG0" | tee -a ~/optena/logs_temporal_main.txt
