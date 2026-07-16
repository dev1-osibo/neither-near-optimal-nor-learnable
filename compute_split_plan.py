"""
READ-ONLY planning script. Does NOT touch the live instance or the live
progress file. Operates only on the LOCAL BACKUP COPY of progress_worker_0.json
to compute exactly which run_ids move to a new progress_worker_1.json under a
2-way split, using the ACTUAL generate_all_runs()/get_worker_runs() functions
imported directly from train_rl_checkpointed.py (no reimplementation, so zero
risk of a transcription mismatch with the live splitting logic).
"""
import sys
import os
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the exact same functions the live script uses
from train_rl_checkpointed import generate_all_runs, get_worker_runs

BACKUP_PROGRESS_FILE = os.path.join(
    os.path.dirname(__file__), "backups", "pre_split_20260704_205246",
    "checkpoints", "progress_worker_0.json"
)

def main():
    all_runs = generate_all_runs()
    print(f"Total runs (should be 240): {len(all_runs)}")
    assert len(all_runs) == 240, "SAFETY CHECK FAILED: expected 240 total runs"

    worker0_new = get_worker_runs(all_runs, 0, 2)
    worker1_new = get_worker_runs(all_runs, 1, 2)
    print(f"Worker 0 new slice size: {len(worker0_new)}")
    print(f"Worker 1 new slice size: {len(worker1_new)}")
    assert len(worker0_new) + len(worker1_new) == 240
    assert len(worker0_new) == len(worker1_new) == 120

    worker0_ids = {r["run_id"] for r in worker0_new}
    worker1_ids = {r["run_id"] for r in worker1_new}
    assert worker0_ids.isdisjoint(worker1_ids), "SAFETY CHECK FAILED: overlap between worker slices"
    assert worker0_ids | worker1_ids == {r["run_id"] for r in all_runs}, "SAFETY CHECK FAILED: union mismatch"

    with open(BACKUP_PROGRESS_FILE) as f:
        backup = json.load(f)

    completed = backup["completed"]
    results = backup["results"]
    in_progress = backup["in_progress"]

    print(f"\nBackup file: completed={len(completed)}, in_progress={in_progress}, results={len(results)}")

    # Sanity: every completed run_id must exist in the global run set
    all_ids = {r["run_id"] for r in all_runs}
    unknown = set(completed) - all_ids
    assert not unknown, f"SAFETY CHECK FAILED: unknown completed run_ids: {unknown}"

    # Split completed run_ids by new ownership
    completed_stay_with_0 = [rid for rid in completed if rid in worker0_ids]
    completed_move_to_1 = [rid for rid in completed if rid in worker1_ids]

    assert len(completed_stay_with_0) + len(completed_move_to_1) == len(completed)

    print(f"\nCompleted runs staying in worker 0's file: {len(completed_stay_with_0)}")
    print(f"Completed runs to MOVE into new worker 1 file: {len(completed_move_to_1)}")
    print("\nRun IDs moving to worker 1:")
    for rid in completed_move_to_1:
        print(" ", rid)

    # Where does the in_progress run land?
    if in_progress:
        owner = "worker0" if in_progress in worker0_ids else ("worker1" if in_progress in worker1_ids else "UNKNOWN")
        print(f"\nIn-progress run '{in_progress}' will be owned by: {owner}")
        assert owner != "UNKNOWN", "SAFETY CHECK FAILED: in-progress run not in any worker's slice"

    # Build the results subset that must move to worker1's results list
    results_by_id = {r["run_id"]: r for r in results if "run_id" in r}
    moved_results = [results_by_id[rid] for rid in completed_move_to_1 if rid in results_by_id]
    assert len(moved_results) == len(completed_move_to_1), "SAFETY CHECK FAILED: missing result for a moved run_id"

    print(f"\nResult entries to move: {len(moved_results)} (matches completed_move_to_1 count: {len(moved_results) == len(completed_move_to_1)})")

    # Write the PROPOSED new files to a local staging dir (NOT the live instance, NOT even a real deploy path)
    staging_dir = os.path.join(os.path.dirname(__file__), "split_staging")
    os.makedirs(staging_dir, exist_ok=True)

    new_worker0_progress = {
        "completed": completed_stay_with_0,
        "in_progress": in_progress if in_progress in worker0_ids else None,
        "results": [results_by_id[rid] for rid in completed_stay_with_0 if rid in results_by_id],
    }
    new_worker1_progress = {
        "completed": completed_move_to_1,
        "in_progress": in_progress if in_progress in worker1_ids else None,
        "results": moved_results,
    }

    with open(os.path.join(staging_dir, "progress_worker_0_PROPOSED.json"), "w") as f:
        json.dump(new_worker0_progress, f, indent=2)
    with open(os.path.join(staging_dir, "progress_worker_1_PROPOSED.json"), "w") as f:
        json.dump(new_worker1_progress, f, indent=2)

    # Final cross-check: total completed count preserved exactly, no loss no duplication
    total_after = len(new_worker0_progress["completed"]) + len(new_worker1_progress["completed"])
    assert total_after == len(completed), f"SAFETY CHECK FAILED: {total_after} != {len(completed)}"
    assert set(new_worker0_progress["completed"]) | set(new_worker1_progress["completed"]) == set(completed)
    assert set(new_worker0_progress["completed"]) & set(new_worker1_progress["completed"]) == set()

    print("\n✓ ALL SAFETY CHECKS PASSED")
    print(f"✓ Proposed files written to: {staging_dir}")
    print(f"✓ Total completed preserved: {total_after} == {len(completed)}")
    print("✓ No overlap, no loss, no duplication between the two proposed worker files")


if __name__ == "__main__":
    main()
