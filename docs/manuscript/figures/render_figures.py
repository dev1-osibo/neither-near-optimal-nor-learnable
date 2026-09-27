"""Render the eight article figures from the published compact result dataset.

The renderer does not run an environment, policy, optimizer, bootstrap, or
inferential test, and it does not select outcomes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_DATA = HERE / "publication_results.json"
EXPECTED_FIGURE_DATA_SHA256 = (
    "db71313e00e1b4e6c03b0f373626e4b531b2589c761d5ddcc6ba83208be44903"
)

INK = "#202A35"
MUTED = "#586575"
GRID = "#DCE2E7"
ELIGIBLE = "#087E72"
INELIGIBLE = "#B8C1C9"
FAIL = "#BB514D"
RECOVERY = "#2875A8"
REGRESSION = "#A06122"
SEED_COLORS = {
    42: "#2875A8",
    123: "#D17A22",
    456: "#087E72",
    789: "#9B59B6",
    1024: "#BB514D",
}
BUDGET_COLORS = {1_000_000: "#6A7785", 2_000_000: "#2875A8", 5_000_000: "#BB514D"}
EXPECTED_PYTHON_VERSION = (3, 14, 4)
EXPECTED_NUMPY_VERSION = "2.4.4"
EXPECTED_MATPLOTLIB_VERSION = "3.10.8"


def check_figure_environment() -> None:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Figure renderer requires CPython {'.'.join(map(str, EXPECTED_PYTHON_VERSION))}; "
            f"found {'.'.join(map(str, actual_python))}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"Figure renderer requires NumPy {EXPECTED_NUMPY_VERSION}; found {np.__version__}"
        )
    if matplotlib.__version__ != EXPECTED_MATPLOTLIB_VERSION:
        raise RuntimeError(
            "Figure renderer requires Matplotlib "
            f"{EXPECTED_MATPLOTLIB_VERSION}; found {matplotlib.__version__}"
        )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    svg_path = HERE / f"{stem}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Date": None,
            "Creator": "Data-Centre Energy-Control Claim Qualification Benchmark",
        },
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    with svg_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
    plt.close(fig)


def load_figure_data() -> dict:
    raw = FIGURE_DATA.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_FIGURE_DATA_SHA256:
        raise RuntimeError(f"Publication figure-data digest mismatch: {actual}")
    data = json.loads(raw)
    if len(data["weeks"]) != 103:
        raise RuntimeError("Publication figure data must contain all 103 weeks")
    if len(data["primary_seed_traces"]) != 20 or len(data["budget_seed_traces"]) != 90:
        raise RuntimeError("Publication figure data population is incomplete")
    return data


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "data-centre-energy-control-claim-qualification",
        }
    )


def evidence_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 2.85))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 2.85)
    ax.axis("off")
    boxes = [
        (0.18, "Causal split\ntrain-only statistics"),
        (2.47, "Common replay\nterminal settlement"),
        (4.76, "All-seed service\nqualification"),
        (7.05, "Complete family\nplus multiplicity"),
        (9.34, "Licensed\nclaim or no claim"),
    ]
    for i, (x, label) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 1.34), 1.88, 0.9,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="#E7F2F0" if i in (2, 4) else "#EFF2F5",
            edgecolor="none",
        )
        ax.add_patch(patch)
        ax.text(x + 0.94, 1.79, label, ha="center", va="center", fontsize=10)
        if i < 4:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.98, 1.79), (boxes[i + 1][0] - 0.1, 1.79),
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=MUTED,
                )
            )
    ax.text(
        5.7, 0.7,
        "Full-episode hindsight optimum: same modeled accounting, privileged future information; diagnostic only",
        ha="center", va="center", fontsize=9.4, color=MUTED,
    )
    ax.plot([1.2, 10.2], [1.12, 1.12], color=GRID, linewidth=1)
    save(fig, "fig1_qualification_logic")


def primary_qualification(figure_data: dict) -> None:
    counts = figure_data["primary_counts"]
    eligible = [counts["individual_policy_eligible"], counts["all_five_seed_slot_eligible"]]
    total = [counts["individual_policy_total"], counts["all_five_seed_slot_total"]]
    labels = ["Individual policies", "All-five-seed slots"]
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    y = [1, 0]
    for yy, n, d in zip(y, eligible, total, strict=True):
        ax.barh(yy, 1, color=INELIGIBLE, height=0.47)
        ax.barh(yy, n / d, color=ELIGIBLE, height=0.47)
        ax.text(n / d + 0.015, yy, f"{n}/{d}", va="center", fontsize=11, fontweight="medium")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share meeting the applicable service screen")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig2_primary_qualification")


def extension_qualification(figure_data: dict) -> None:
    rows = figure_data["extension_qualification_counts"]
    order = [
        ("History probe", "PPO"), ("History probe", "RecurrentPPO"),
        ("Budget probe", "PPO"), ("Budget probe", "RecurrentPPO"),
        ("Budget probe", "SAC"),
    ]
    expected = [(9, 11), (2, 18), (17, 3), (9, 11), (13, 7)]
    counts = {
        (row["probe"], row["algorithm"]): (row["eligible"], row["ineligible"])
        for row in rows
    }
    observed = [counts[key] for key in order]
    assert observed == expected, observed
    assert sum(a for a, _ in observed) == 50
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = list(range(len(order) - 1, -1, -1))
    for yy, (a, b) in zip(y, observed, strict=True):
        ax.barh(yy, a, height=0.54, color=ELIGIBLE)
        ax.barh(yy, b, left=a, height=0.54, color=INELIGIBLE)
        ax.text(a / 2, yy, str(a), ha="center", va="center", color="white" if a >= 4 else INK)
        ax.text(a + b / 2, yy, str(b), ha="center", va="center", color=INK)
    ax.set_yticks(y, [f"{probe} · {algo}" for probe, algo in order])
    ax.set_xlim(0, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Fixed policy artifacts (five seeds per declared slot)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        [Rectangle((0, 0), 1, 1, color=ELIGIBLE), Rectangle((0, 0), 1, 1, color=INELIGIBLE)],
        ["Service-eligible", "Ineligible"], frameon=False, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
    )
    fig.tight_layout()
    save(fig, "fig5_extension_qualification")


def budget_transitions(figure_data: dict) -> None:
    records = figure_data["budget_classification_transitions"]
    assert len(records) == 12
    status = dict(records)
    assert Counter(status.values()) == Counter(
        {"COMPLIANCE_RECOVERY": 1, "COMPLIANCE_FAILURE_PERSISTS": 9, "COMPLIANCE_REGRESSION": 2}
    )
    configs = ["all_sources", "grid_solar_wind_battery"]
    algos = ["PPO", "SAC", "RecurrentPPO"]
    budgets = ["2000000", "5000000"]
    colors = {
        "COMPLIANCE_RECOVERY": RECOVERY,
        "COMPLIANCE_FAILURE_PERSISTS": FAIL,
        "COMPLIANCE_REGRESSION": REGRESSION,
    }
    short = {
        "COMPLIANCE_RECOVERY": "Recovery",
        "COMPLIANCE_FAILURE_PERSISTS": "Persistent failure",
        "COMPLIANCE_REGRESSION": "Regression",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 3.65), sharey=True)
    for ax, config in zip(axes, configs, strict=True):
        for row, algo in enumerate(algos):
            for col, budget in enumerate(budgets):
                key = f"{config}/{algo}/{budget}-vs-1000000"
                code = status[key]
                ax.add_patch(Rectangle((col, 2 - row), 0.94, 0.86, facecolor=colors[code], edgecolor="none"))
                ax.text(col + 0.47, 2 - row + 0.43, short[code],
                        ha="center", va="center", color="white", fontsize=9)
        ax.set_xlim(-0.03, 1.97)
        ax.set_ylim(-0.07, 2.93)
        ax.set_xticks([0.47, 1.47], ["2M vs 1M", "5M vs 1M"])
        ax.set_yticks([2.43, 1.43, 0.43], algos)
        ax.set_title(config.replace("_", " "), pad=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig6_budget_transitions")


def primary_weekly_cost_by_seed(figure_data: dict) -> None:
    rows = figure_data["primary_seed_traces"]
    weeks = np.arange(1, 104)
    algorithms = ["PPO", "SAC", "TD3", "A2C"]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), sharex=True)
    for ax, algorithm in zip(axes.flat, algorithms, strict=True):
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: row["seed"],
        )
        if len(selected) != 5:
            raise RuntimeError(f"missing primary seed traces for {algorithm}")
        for row in selected:
            seed = int(row["seed"])
            ax.plot(
                weeks,
                row["weekly_cost_gap_to_rulebased_pct"],
                color=SEED_COLORS[seed],
                linewidth=1.05,
                alpha=0.82,
                label=f"seed {seed}",
            )
        ax.axhline(0, color=INK, linewidth=0.8, alpha=0.75)
        ax.set_title(algorithm)
        ax.grid(color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Weekly cost gap to RuleBased (%)")
    axes[1, 0].set_ylabel("Weekly cost gap to RuleBased (%)")
    axes[1, 0].set_xlabel("Held-out week (1–103)")
    axes[1, 1].set_xlabel("Held-out week (1–103)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_primary_weekly_cost_by_seed")


def primary_cumulative_service_by_seed(figure_data: dict) -> None:
    rows = figure_data["primary_seed_traces"]
    weeks = np.arange(1, 104)
    algorithms = ["PPO", "SAC", "TD3", "A2C"]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), sharex=True)
    for ax, algorithm in zip(axes.flat, algorithms, strict=True):
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: row["seed"],
        )
        for row in selected:
            seed = int(row["seed"])
            ax.plot(
                weeks,
                np.cumsum(row["weekly_sla_violations_all_configurations"]),
                color=SEED_COLORS[seed],
                linewidth=1.25,
                alpha=0.88,
                drawstyle="steps-post",
                label=f"seed {seed}",
            )
        ax.set_title(algorithm)
        ax.grid(color=GRID, linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Cumulative service violations\nacross 12 configurations")
    axes[1, 0].set_ylabel("Cumulative service violations\nacross 12 configurations")
    axes[1, 0].set_xlabel("Held-out week (1–103)")
    axes[1, 1].set_xlabel("Held-out week (1–103)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig4_primary_cumulative_service_by_seed")


def _budget_panel_data(figure_data: dict, configuration: str, algorithm: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in figure_data["budget_seed_traces"]:
        if row["configuration"] == configuration and row["algorithm"] == algorithm:
            grouped[int(row["requested_timesteps"])].append(row)
    if {budget: len(records) for budget, records in grouped.items()} != {
        1_000_000: 5, 2_000_000: 5, 5_000_000: 5
    }:
        raise RuntimeError(f"incomplete budget panel: {configuration}/{algorithm}")
    return grouped


def budget_weekly_cost_snapshots(figure_data: dict) -> None:
    weeks = np.arange(1, 104)
    configs = ["all_sources", "grid_solar_wind_battery"]
    algorithms = ["PPO", "SAC", "RecurrentPPO"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.5), sharex=True)
    for row_index, config in enumerate(configs):
        for col_index, algorithm in enumerate(algorithms):
            ax = axes[row_index, col_index]
            grouped = _budget_panel_data(figure_data, config, algorithm)
            for budget in (1_000_000, 2_000_000, 5_000_000):
                values = []
                for record in sorted(grouped[budget], key=lambda item: item["seed"]):
                    series = np.asarray(record["weekly_terminal_adjusted_cost_usd"], dtype=float)
                    values.append(series)
                    ax.plot(weeks, series, color=BUDGET_COLORS[budget], linewidth=0.55, alpha=0.18)
                ax.plot(
                    weeks,
                    np.mean(values, axis=0),
                    color=BUDGET_COLORS[budget],
                    linewidth=1.65,
                    label=f"{budget // 1_000_000}M seed mean",
                )
            ax.set_title(f"{config.replace('_', ' ')} · {algorithm}", fontsize=10)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.75)
            ax.spines[["top", "right"]].set_visible(False)
            if col_index == 0:
                ax.set_ylabel("Weekly terminal-adjusted cost (USD)")
            if row_index == 1:
                ax.set_xlabel("Held-out week (1–103)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig7_budget_weekly_cost_snapshots")


def budget_cumulative_service_snapshots(figure_data: dict) -> None:
    weeks = np.arange(1, 104)
    configs = ["all_sources", "grid_solar_wind_battery"]
    algorithms = ["PPO", "SAC", "RecurrentPPO"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.5), sharex=True)
    for row_index, config in enumerate(configs):
        for col_index, algorithm in enumerate(algorithms):
            ax = axes[row_index, col_index]
            grouped = _budget_panel_data(figure_data, config, algorithm)
            for budget in (1_000_000, 2_000_000, 5_000_000):
                values = []
                for record in sorted(grouped[budget], key=lambda item: item["seed"]):
                    series = np.cumsum(record["weekly_sla_violations"])
                    values.append(series)
                    ax.plot(
                        weeks,
                        series,
                        color=BUDGET_COLORS[budget],
                        linewidth=0.75,
                        alpha=0.24,
                        drawstyle="steps-post",
                    )
                ax.plot(
                    weeks,
                    np.mean(values, axis=0),
                    color=BUDGET_COLORS[budget],
                    linewidth=1.7,
                    drawstyle="steps-post",
                    label=f"{budget // 1_000_000}M seed mean",
                )
            ax.set_title(f"{config.replace('_', ' ')} · {algorithm}", fontsize=10)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.75)
            ax.spines[["top", "right"]].set_visible(False)
            if col_index == 0:
                ax.set_ylabel("Cumulative service violations per seed")
            if row_index == 1:
                ax.set_xlabel("Held-out week (1–103)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig8_budget_cumulative_service_snapshots")


def main() -> None:
    check_figure_environment()
    setup()
    figure_data = load_figure_data()
    evidence_flow()
    primary_qualification(figure_data)
    extension_qualification(figure_data)
    budget_transitions(figure_data)
    primary_weekly_cost_by_seed(figure_data)
    primary_cumulative_service_by_seed(figure_data)
    budget_weekly_cost_snapshots(figure_data)
    budget_cumulative_service_snapshots(figure_data)
    print("Rendered eight figures from digest-verified complete publication populations.")


if __name__ == "__main__":
    main()
