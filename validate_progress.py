"""Read-only validation of progress_worker_0.json. Does not modify anything."""
import json

with open("/home/ubuntu/optena/checkpoints/progress_worker_0.json") as f:
    d = json.load(f)

print("completed count:", len(d["completed"]))
print("in_progress:", d["in_progress"])
print("results count:", len(d["results"]))
print("completed == unique:", len(d["completed"]) == len(set(d["completed"])))
print()
print("completed run_ids:")
for r in d["completed"]:
    print(" ", r)
print()
# Cross-check: every completed run_id has a matching result entry
result_ids = {r["run_id"] for r in d["results"] if "run_id" in r}
missing = set(d["completed"]) - result_ids
extra = result_ids - set(d["completed"])
print("completed without matching result:", missing)
print("results without matching completed entry:", extra)
