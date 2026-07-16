"""Print EDA 08 results summary."""
import json

with open("/home/ubuntu/optena/results/eda_deep_cross_variations_results.json") as f:
    r = json.load(f)

print("Keys:", list(r.keys()))
print()

print("=== GRANGER CAUSALITY ===")
for target, preds in r.get("granger_causality", {}).items():
    print(f"  {target}:")
    for pred, vals in preds.items():
        print(f"    {pred} -> {vals['causes']} (p={vals['min_p_value']:.2e}, lag={vals['best_lag']}h)")

print()
print("=== NON-LINEAR INTERACTIONS ===")
for pair, vals in r.get("nonlinear_interactions", {}).items():
    nl = "NON-LINEAR" if vals["is_nonlinear"] else "linear"
    print(f"  {pair}: R2_lin={vals['r2_linear']:.3f} R2_poly={vals['r2_polynomial']:.3f} [{nl}]")

print()
print("=== FEATURE IMPORTANCE ===")
for target, fi in r.get("feature_importance", {}).items():
    print(f"  {target}: R2={fi['r2']:.4f}, MAPE={fi['mape']:.1f}%")
    for feat, imp in fi["top_5"]:
        print(f"    {feat}: {imp:.4f}")

print()
print("=== ABLATION STUDY ===")
if "ablation_study" in r:
    abl = r["ablation_study"]
    print(f"  Target: {abl['target']}, Baseline R2: {abl['baseline_r2']:.4f}")
    for group, vals in abl["groups"].items():
        print(f"  Without {group}: R2={vals['r2_without']:.4f} (drop: {vals['r2_drop_pct']:.1f}%)")

print()
print("=== CONDITIONAL PRICE DISTRIBUTIONS ===")
for cond, vals in r.get("conditional_distributions", {}).items():
    print(f"  {cond:15s}: mean=${vals['price_mean']:.1f}, p95=${vals['price_p95']:.0f}")

print()
print("=== REGIMES ===")
if "regimes" in r:
    print(f"  Names: {r['regimes']['regime_names']}")
    print(f"  Sizes: {r['regimes']['sizes']}")
