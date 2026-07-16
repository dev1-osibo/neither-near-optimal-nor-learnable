"""Read-only status check across both worker progress files."""
import json

with open("/home/ubuntu/optena/checkpoints/progress_worker_0.json") as f:
    w0 = json.load(f)
with open("/home/ubuntu/optena/checkpoints/progress_worker_1.json") as f:
    w1 = json.load(f)

print("Worker 0: completed =", len(w0["completed"]), " in_progress =", w0["in_progress"])
print("Worker 1: completed =", len(w1["completed"]), " in_progress =", w1["in_progress"])

total = len(w0["completed"]) + len(w1["completed"])
print(f"\nTOTAL completed: {total} / 240 ({total/240*100:.1f}%)")

overlap = set(w0["completed"]) & set(w1["completed"])
print("Overlap check (must be empty):", overlap)

dup_check_0 = len(w0["completed"]) == len(set(w0["completed"]))
dup_check_1 = len(w1["completed"]) == len(set(w1["completed"]))
print("Worker 0 no internal dupes:", dup_check_0)
print("Worker 1 no internal dupes:", dup_check_1)
