"""Validate the compact result populations published with the article."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


RESULTS = Path(__file__).with_name("publication_results.json")


def validate() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert len(data["weeks"]) == 103
    assert len(data["primary_configurations"]) == 12
    assert len(data["primary_seed_traces"]) == 20
    assert len(data["budget_seed_traces"]) == 90
    assert data["primary_counts"] == {
        "all_five_seed_slot_eligible": 4,
        "all_five_seed_slot_total": 48,
        "individual_policy_eligible": 136,
        "individual_policy_total": 240,
    }

    extension = data["extension_qualification_counts"]
    assert len(extension) == 5
    assert sum(row["eligible"] + row["ineligible"] for row in extension) == 100
    assert sum(row["eligible"] for row in extension) == 50

    transitions = data["budget_classification_transitions"]
    assert len(transitions) == 12
    assert Counter(transitions.values()) == Counter(
        {
            "COMPLIANCE_RECOVERY": 1,
            "COMPLIANCE_FAILURE_PERSISTS": 9,
            "COMPLIANCE_REGRESSION": 2,
        }
    )


if __name__ == "__main__":
    validate()
    print("Publication result populations and reported totals are complete.")
