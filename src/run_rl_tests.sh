#!/usr/bin/env bash
# Sequential runner for the RL significance + adversarial suites.
# Sequential (not parallel) so the three multiprocessing jobs don't
# oversubscribe the 8 vCPUs. Logs everything with timestamps.
set -u
cd ~/optena
source venv/bin/activate

LOG=~/optena/logs_rl_tests.txt
echo "=== RL TEST SUITE START $(date -u) ===" | tee "$LOG"

echo ">>> [1/3] Significance FULL window (200 ep) $(date -u)" | tee -a "$LOG"
python src/rl_significance.py --split full --n-episodes 200 --n-procs 7 >> "$LOG" 2>&1
echo ">>> [1/3] done rc=$? $(date -u)" | tee -a "$LOG"

echo ">>> [2/3] Significance TEST window / holdout (200 ep) $(date -u)" | tee -a "$LOG"
python src/rl_significance.py --split test --n-episodes 200 --n-procs 7 >> "$LOG" 2>&1
echo ">>> [2/3] done rc=$? $(date -u)" | tee -a "$LOG"

echo ">>> [3/3] Adversarial battery PPO (100 ep) $(date -u)" | tee -a "$LOG"
python src/rl_adversarial.py --algo PPO --n-episodes 100 --n-procs 7 >> "$LOG" 2>&1
echo ">>> [3/3] done rc=$? $(date -u)" | tee -a "$LOG"

echo "=== RL TEST SUITE COMPLETE $(date -u) ===" | tee -a "$LOG"
