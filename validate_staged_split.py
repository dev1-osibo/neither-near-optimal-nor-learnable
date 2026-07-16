"""Read-only validation of the two PROPOSED progress files in split_staging/.
Does not touch checkpoints/ (the live directory) at all."""
import json
import os

STAGING = "/home/ubuntu/optena/split_staging"

with open(os.path.join(STAGING, "progress_worker_0_PROPOSED.json")) as f:
    w0 = json.load(f)
with open(os.path.join(STAGING, "progress_worker_1_PROPOSED.json")) as f:
    w1 = json.load(f)

print("=== Worker 0 PROPOSED ===")
print("completed:", len(w0["completed"]))
print("in_progress:", w0["in_progress"])
print("results:", len(w0["results"]))

print("\n=== Worker 1 PROPOSED ===")
print("completed:", len(w1["completed"]))
print("in_progress:", w1["in_progress"])
print("results:", len(w1["results"]))

overlap = set(w0["completed"]) & set(w1["completed"])
print("\nOverlap between the two (must be empty):", overlap)
print("Combined completed:", len(w0["completed"]) + len(w1["completed"]), "(should equal 31)")

# results integrity
for label, d in [("worker0", w0), ("worker1", w1)]:
    ids_completed = set(d["completed"])
    ids_results = {r["run_id"] for r in d["results"] if "run_id" in r}
    print(f"{label}: completed==results ids match:", ids_completed == ids_results)
