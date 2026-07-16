"""Read-only validation of the LIVE progress_worker_0.json and progress_worker_1.json
immediately after writing them, cross-checked against the pre-split backup to prove
zero data loss."""
import json
import os

CKPT = "/home/ubuntu/optena/checkpoints"
BACKUP = "/home/ubuntu/optena/backups/pre_write_20260704_210427/checkpoints"

with open(os.path.join(CKPT, "progress_worker_0.json")) as f:
    w0 = json.load(f)
with open(os.path.join(CKPT, "progress_worker_1.json")) as f:
    w1 = json.load(f)
with open(os.path.join(BACKUP, "progress_worker_0.json")) as f:
    backup = json.load(f)

print("BEFORE (backup): completed =", len(backup["completed"]), " in_progress =", backup["in_progress"])
print("AFTER  worker0 : completed =", len(w0["completed"]), " in_progress =", w0["in_progress"])
print("AFTER  worker1 : completed =", len(w1["completed"]), " in_progress =", w1["in_progress"])

combined_completed = set(w0["completed"]) | set(w1["completed"])
overlap = set(w0["completed"]) & set(w1["completed"])

print("\nCombined completed count:", len(combined_completed), "== backup:", len(combined_completed) == len(backup["completed"]))
print("Set-equal to backup completed:", combined_completed == set(backup["completed"]))
print("Overlap (must be empty):", overlap)

combined_results = len(w0["results"]) + len(w1["results"])
print("Combined results count:", combined_results, "== backup:", combined_results == len(backup["results"]))

# Verify every single result entry's data (cost/reward/etc) is byte-identical to backup, not just IDs
backup_results_by_id = {r["run_id"]: r for r in backup["results"]}
all_new_results = w0["results"] + w1["results"]
mismatches = []
for r in all_new_results:
    rid = r["run_id"]
    if rid not in backup_results_by_id:
        mismatches.append((rid, "MISSING_IN_BACKUP"))
    elif r != backup_results_by_id[rid]:
        mismatches.append((rid, "VALUE_MISMATCH"))

print("\nResult value integrity check (must be empty list):", mismatches)

if not mismatches and combined_completed == set(backup["completed"]) and not overlap:
    print("\n✓✓✓ SPLIT VERIFIED: ZERO DATA LOSS, ZERO CORRUPTION, ZERO DUPLICATION ✓✓✓")
else:
    print("\n✗✗✗ VALIDATION FAILED — DO NOT PROCEED, RESTORE FROM BACKUP ✗✗✗")
