from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "manuscript" / "figures"
RESULTS = FIGURES / "publication_results.json"


def load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_metadata_and_creator() -> None:
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert metadata["title"] == "Data-Centre Energy-Control Claim Qualification Benchmark"
    assert metadata["creators"] == [{"name": "Osibo, Babasola"}]
    assert metadata["version"] == "0.2.1"


def test_complete_primary_population() -> None:
    data = load_results()
    assert len(data["weeks"]) == 103
    assert len(data["primary_configurations"]) == 12
    assert len(data["primary_seed_traces"]) == 20
    assert data["primary_counts"] == {
        "all_five_seed_slot_eligible": 4,
        "all_five_seed_slot_total": 48,
        "individual_policy_eligible": 136,
        "individual_policy_total": 240,
    }


def test_complete_extension_population() -> None:
    data = load_results()
    assert len(data["budget_seed_traces"]) == 90
    counts = data["extension_qualification_counts"]
    assert sum(row["eligible"] + row["ineligible"] for row in counts) == 100
    assert sum(row["eligible"] for row in counts) == 50
    assert sum(row["ineligible"] for row in counts) == 50


def test_budget_transition_totals() -> None:
    transitions = load_results()["budget_classification_transitions"]
    assert len(transitions) == 12
    assert Counter(transitions.values()) == Counter(
        {
            "COMPLIANCE_RECOVERY": 1,
            "COMPLIANCE_FAILURE_PERSISTS": 9,
            "COMPLIANCE_REGRESSION": 2,
        }
    )


def test_all_eight_figures_are_published_in_both_formats() -> None:
    assert len(list(FIGURES.glob("fig*.png"))) == 8
    assert len(list(FIGURES.glob("fig*.svg"))) == 8


def test_public_text_has_no_numbered_development_labels_or_placeholders() -> None:
    patterns = [
        re.compile(r"task[ _-]*\d+", re.IGNORECASE),
        re.compile(r"\b(?:TB" + r"D|TO" + r"DO)\b"),
        re.compile(r"\[\[PLACE" + r"HOLDER\]\]", re.IGNORECASE),
        re.compile(r"review " + r"branch", re.IGNORECASE),
    ]
    readable = {".md", ".json", ".cff", ".py", ".txt", ".ini"}
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if path.is_file() and path.suffix.lower() in readable:
            text = path.read_text(encoding="utf-8")
            assert not any(pattern.search(text) for pattern in patterns), path


def test_package_has_no_research_operation_directories() -> None:
    excluded = {"artifacts", "config", "contracts", "protocol_addenda", "scripts", "src"}
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        relative_parts = {part.lower() for part in path.relative_to(ROOT).parts}
        assert not relative_parts.intersection(excluded), path
