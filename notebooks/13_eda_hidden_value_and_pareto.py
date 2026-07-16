"""
EDA 13: Hidden Value Discovery & Pareto Frontier Analysis
===========================================================
Questions:
1. Where is the $1.07M/yr gap hiding? (omniscient vs heuristic breakdown)
2. The cost-carbon Pareto frontier — can we have BOTH?
3. Time-of-use patterns the heuristic misses
4. Inter-day optimization (multi-day battery cycling)
5. Seasonal strategy switching — does the optimal policy change by season?
6. Price spike capture rate — how many spikes do we dodge vs miss?
7. The "perfect storm" scenarios — worst hours for each strategy
8. Revenue from grid services (frequency response, demand response)
9. Value of EACH additional forecast hour (diminishing returns curve)
10. Regional portfolio effect — diversification value across ERCOT+CAISO
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.expanduser("~/optena/data")
RESULTS_DIR = os.path.expanduser("~/optena/results")
results = {}

print("=" * 70)
print("EDA 13: HIDDEN VALUE & PARETO FRONTIER")
print("=" * 70)

# Load and setup (same as EDA 12)
merged = pd.read_csv(os.path.join(DATA_DIR, "merged_enriched_2020_2025.csv"))
merged["timestamp"] = pd.to_datetime(merged["timestamp"])

ercot = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_ERCOT_2020_2025.csv"))
ercot["timestamp"] = pd.to_datetime(ercot["timestamp"])

caiso = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_CAISO_2020_2025.csv"))
caiso["timestamp"] = pd.to_datetime(caiso["timestamp"], utc=True)
caiso["timestamp"] = caiso["timestamp"].dt.tz_localize(None)

gas = pd.read_csv(os.path.join(DATA_DIR, "real_gas_henry_hub_daily_2020_2025.csv"))
gas["date"] = pd.to_datetime(gas["date"])

SCALE = 10
PANEL_AREA = 5556
merged["solar_gen_kw"] = (merged["shortwave_radiation"] * PANEL_AREA * 0.18 * 0.85) / 1000

def wind_power(speed, rated=2000):
    p = np.zeros_like(speed, dtype=float)
    mask = (speed >= 3.5) & (speed < 12)
    p[mask] = rated * ((speed[mask] - 3.5) / 8.5) ** 3
    p[(speed >= 12) & (speed <= 25)] = rated
    return p

merged["wind_gen_kw"] = wind_power(merged["wind_speed_10m"].values)

df = merged.merge(ercot[["timestamp", "lmp_price_usd_mwh"]], on="timestamp", how="left")
df["lmp_price_usd_mwh"] = df["lmp_price_usd_mwh"].ffill().bfill()
df = df.dropna(subset=["lmp_price_usd_mwh"]).copy()

gas_map = gas.set_index(gas["date"].dt.strftime("%Y-%m-%d"))["gas_price_usd_mmbtu"].to_dict()
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
df["gas_cost_mwh"] = df["date_str"].map(gas_map).apply(lambda x: x / 0.11723 if pd.notna(x) else 29.3)

facility_demand = df["total_facility_kw"].values * SCALE
grid_price = df["lmp_price_usd_mwh"].values / 1000
solar = df["solar_gen_kw"].values * 5
wind = df["wind_gen_kw"].values * 2.5
gas_cost = df["gas_cost_mwh"].values / 1000
GRID_CARBON = df["carbon_intensity_gco2_kwh"].values / 1000
GAS_CARBON = 0.00041

BATTERY_CAP = 20000
BATTERY_RATE = 10000
BATTERY_EFF = 0.90
GAS_CAP = 2000
years = len(df) / 8760

print(f"  Setup complete: {len(df):,} hours, 10MW facility")

# ============================================================
# 1. WHERE IS THE HIDDEN VALUE? (Decompose the gap)
# ============================================================
print("\n[1] Decomposing the gap to optimal...")

# The heuristic misses value in several places. Let's quantify each:
# A) Battery timing: heuristic uses 24h window, optimal uses FULL horizon
# B) Workload shifting: heuristic uses 12h blocks, optimal can shift days
# C) Gas timing: heuristic just checks threshold, optimal is more selective
# D) Renewable curtailment: when battery is full and renewables exceed demand

# A) Battery: optimal vs our 24h-window heuristic
# Perfect battery: always charge at day's absolute cheapest, discharge at absolute expensive
battery_value_perfect = 0.0
battery_value_heuristic = 0.0

for day_start in range(0, len(grid_price) - 24, 24):
    day_prices = grid_price[day_start:day_start+24]
    
    # Perfect: buy at min, sell at max (one cycle per day)
    if len(day_prices) == 24:
        min_p = day_prices.min()
        max_p = day_prices.max()
        perfect_profit = (max_p - min_p) * BATTERY_CAP * BATTERY_EFF / 1000  # $k
        battery_value_perfect += perfect_profit
        
        # Heuristic: buy at p25 of 24h window, sell at p75
        p25 = np.percentile(day_prices, 25)
        p75 = np.percentile(day_prices, 75)
        heuristic_profit = max(0, (p75 - p25)) * BATTERY_CAP * BATTERY_EFF / 1000
        battery_value_heuristic += heuristic_profit

print(f"  Battery arbitrage (annual):")
print(f"    Perfect timing: ${battery_value_perfect/years:,.0f}/yr")
print(f"    24h heuristic: ${battery_value_heuristic/years:,.0f}/yr")
print(f"    Gap: ${(battery_value_perfect - battery_value_heuristic)/years:,.0f}/yr")

# B) Workload shifting value at different window sizes
print(f"\n  Workload deferral window sensitivity:")
deferrable = facility_demand * 0.30
fixed_demand = facility_demand * 0.70

for window_hours in [4, 12, 24, 48, 168]:
    savings = 0.0
    for block_start in range(0, len(grid_price) - window_hours, window_hours):
        block_end = block_start + window_hours
        block_prices = grid_price[block_start:block_end]
        block_defer = deferrable[block_start:block_end]
        
        # Cost without deferral
        cost_no_defer = np.sum(block_defer * block_prices)
        
        # Cost with deferral (serve in cheapest hours)
        total_energy = block_defer.sum()
        sorted_p = np.sort(block_prices)
        # Fill cheapest hours first
        n_hours_needed = max(1, int(np.ceil(total_energy / facility_demand.max())))
        cheapest_prices = sorted_p[:min(n_hours_needed, len(sorted_p))]
        cost_deferred = total_energy * cheapest_prices.mean() if len(cheapest_prices) > 0 else cost_no_defer
        
        savings += (cost_no_defer - cost_deferred)
    
    print(f"    {window_hours}h window: saves ${savings/years:,.0f}/yr")

# C) Renewable curtailment — how much free energy are we wasting?
excess_renewable = np.maximum(0, (solar + wind) - facility_demand)
curtailed_energy = excess_renewable.sum() / 1000  # MWh
curtailed_value = np.sum(excess_renewable * grid_price) / years  # If we could store/sell it
print(f"\n  Renewable curtailment:")
print(f"    Annual curtailed energy: {curtailed_energy/years:,.0f} MWh/yr")
print(f"    Value if captured: ${curtailed_value:,.0f}/yr")
print(f"    (This is what bigger battery or flexible load captures)")

results["hidden_value_decomposition"] = {
    "battery_gap_annual": float((battery_value_perfect - battery_value_heuristic) / years),
    "curtailed_renewable_mwh": float(curtailed_energy / years),
    "curtailed_value_usd": float(curtailed_value),
}

# ============================================================
# 2. PARETO FRONTIER: COST vs CARBON
# ============================================================
print("\n[2] Cost-Carbon Pareto Frontier...")
print("  Sweeping the alpha weight to map the tradeoff curve")

# Run the coordinated strategy with different cost-vs-carbon weights
# alpha=1.0 = purely minimize cost (ignore carbon)
# alpha=0.0 = purely minimize carbon (ignore cost)

def coordinated_with_alpha(demand, grid_price, solar, wind, gas_cost, grid_carbon, alpha_cost):
    """Run coordinated strategy with cost-carbon weighting."""
    alpha_carbon = 1.0 - alpha_cost
    CARBON_PRICE = 0.10  # $/kg CO2 equivalent for decision-making
    
    total_cost = 0.0
    total_carbon = 0.0
    battery_soc = BATTERY_CAP * 0.5
    
    for t in range(len(demand)):
        rem = demand[t]
        
        # Renewables (always first — free AND clean)
        rem -= min(solar[t], rem)
        rem -= min(wind[t], max(0, rem))
        
        # Battery: discharge when weighted cost is high
        if t + 24 < len(grid_price) and battery_soc > BATTERY_CAP * 0.1 and rem > 0:
            effective_price = alpha_cost * grid_price[t] + alpha_carbon * grid_carbon[t] * CARBON_PRICE
            future_eff = [alpha_cost * grid_price[t+i] + alpha_carbon * grid_carbon[t+i] * CARBON_PRICE 
                         for i in range(min(24, len(grid_price)-t))]
            if effective_price > np.percentile(future_eff, 70):
                discharge = min(rem, BATTERY_RATE, battery_soc * BATTERY_EFF)
                rem -= discharge
                battery_soc -= discharge / BATTERY_EFF
            elif effective_price < np.percentile(future_eff, 30) and battery_soc < BATTERY_CAP * 0.9:
                charge = min(BATTERY_RATE, (BATTERY_CAP - battery_soc) / BATTERY_EFF)
                battery_soc += charge * BATTERY_EFF
                total_cost += charge * grid_price[t]
                total_carbon += charge * grid_carbon[t]
        
        # Gas: use when it has lower WEIGHTED cost than grid
        if rem > 0:
            gas_weighted = alpha_cost * gas_cost[t] + alpha_carbon * GAS_CARBON * CARBON_PRICE
            grid_weighted = alpha_cost * grid_price[t] + alpha_carbon * grid_carbon[t] * CARBON_PRICE
            
            if gas_weighted < grid_weighted:
                gas_used = min(GAS_CAP, rem)
                total_cost += gas_used * gas_cost[t]
                total_carbon += gas_used * GAS_CARBON
                rem -= gas_used
        
        # Grid
        if rem > 0:
            total_cost += rem * grid_price[t]
            total_carbon += rem * grid_carbon[t]
        
        # Free charge
        excess = max(0, solar[t] + wind[t] - demand[t])
        if excess > 0 and battery_soc < BATTERY_CAP:
            fc = min(excess, BATTERY_RATE, (BATTERY_CAP - battery_soc) / BATTERY_EFF)
            battery_soc += fc * BATTERY_EFF
    
    return total_cost / years, total_carbon / years

# Sweep alpha from pure-cost to pure-carbon
alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
pareto_points = []

print(f"\n  {'Alpha(cost)':>12} | {'Annual Cost':>12} | {'Annual Carbon':>14} | Direction")
print(f"  {'-'*12} | {'-'*12} | {'-'*14} | {'-'*20}")

for alpha in alphas:
    cost, carbon = coordinated_with_alpha(
        facility_demand, grid_price, solar, wind, gas_cost, GRID_CARBON, alpha
    )
    pareto_points.append({"alpha": alpha, "cost": cost, "carbon": carbon})
    
    if alpha == 1.0:
        direction = "← Pure cost minimization"
    elif alpha == 0.0:
        direction = "← Pure carbon minimization"
    elif alpha == 0.5:
        direction = "← Balanced"
    else:
        direction = ""
    print(f"  {alpha:>12.1f} | ${cost:>10,.0f} | {carbon:>12,.0f} kg | {direction}")

# Is Pareto improvement possible? (reduce BOTH cost AND carbon vs baseline)
baseline_cost = float(np.sum(facility_demand * grid_price)) / years
baseline_carbon = float(np.sum(facility_demand * GRID_CARBON)) / years

pareto_improvements = [p for p in pareto_points if p["cost"] < baseline_cost and p["carbon"] < baseline_carbon]
print(f"\n  Pareto improvements over grid-only (lower cost AND lower carbon):")
print(f"    {len(pareto_improvements)} out of {len(alphas)} alpha settings achieve both!")
if pareto_improvements:
    best_balanced = min(pareto_improvements, key=lambda p: p["cost"] + p["carbon"] * 0.10)
    print(f"    Best balanced: alpha={best_balanced['alpha']}, "
          f"cost=${best_balanced['cost']:,.0f}, carbon={best_balanced['carbon']:,.0f} kg")
    cost_saving = (baseline_cost - best_balanced["cost"]) / baseline_cost * 100
    carbon_saving = (baseline_carbon - best_balanced["carbon"]) / baseline_carbon * 100
    print(f"    Saves {cost_saving:.1f}% cost AND {carbon_saving:.1f}% carbon simultaneously!")

results["pareto_frontier"] = {
    "points": pareto_points,
    "baseline_cost": float(baseline_cost),
    "baseline_carbon": float(baseline_carbon),
    "n_pareto_improvements": len(pareto_improvements),
}

# ============================================================
# 3. PRICE SPIKE CAPTURE RATE
# ============================================================
print("\n[3] Price Spike Capture Analysis...")
print("  How many extreme price spikes does each strategy dodge?")

# Define spikes at various thresholds
thresholds = [100, 200, 500, 1000, 5000]  # $/MWh

for thresh in thresholds:
    spike_hours = grid_price > (thresh / 1000)
    n_spikes = spike_hours.sum()
    
    if n_spikes == 0:
        continue
    
    # Value at risk during spikes
    spike_cost_grid_only = np.sum(facility_demand[spike_hours] * grid_price[spike_hours])
    # If we discharged battery during ALL spikes:
    battery_could_serve = min(BATTERY_CAP * BATTERY_EFF, facility_demand[spike_hours].mean())
    avoided_per_spike = battery_could_serve * grid_price[spike_hours].mean()
    total_avoidable = avoided_per_spike * n_spikes
    
    print(f"  >${thresh}/MWh spikes: {n_spikes} hours ({n_spikes/len(grid_price)*100:.2f}%)")
    print(f"    Total cost exposure: ${spike_cost_grid_only/years:,.0f}/yr")
    print(f"    Battery could avoid: ${total_avoidable/years:,.0f}/yr (if always fully charged)")

# ============================================================
# 4. SEASONAL STRATEGY VARIATION
# ============================================================
print("\n[4] Seasonal Strategy Variation...")
print("  Does the optimal mix change by season?")

df["month"] = df["timestamp"].dt.month
df["season"] = df["month"].map({12:"Winter", 1:"Winter", 2:"Winter",
                                 3:"Spring", 4:"Spring", 5:"Spring",
                                 6:"Summer", 7:"Summer", 8:"Summer",
                                 9:"Fall", 10:"Fall", 11:"Fall"})

for season in ["Winter", "Spring", "Summer", "Fall"]:
    mask = df["season"].values == season
    s_demand = facility_demand[mask]
    s_price = grid_price[mask]
    s_solar = solar[mask]
    s_wind = wind[mask]
    s_gas = gas_cost[mask]
    
    avg_price = s_price.mean() * 1000
    solar_cf = s_solar.mean() / (5000)  # vs 5MW capacity
    wind_cf = s_wind.mean() / (5000)    # vs 5MW capacity
    renewable_pct = (s_solar.sum() + s_wind.sum()) / s_demand.sum() * 100
    gas_cheaper_pct = (s_gas < s_price).mean() * 100
    
    print(f"\n  {season}:")
    print(f"    Avg price: ${avg_price:.0f}/MWh | Solar CF: {solar_cf*100:.0f}% | Wind CF: {wind_cf*100:.0f}%")
    print(f"    Renewable covers: {renewable_pct:.0f}% of demand")
    print(f"    Gas cheaper than grid: {gas_cheaper_pct:.0f}% of hours")
    
    # Best strategy for this season
    if renewable_pct > 40:
        best = "Maximize renewable capture + battery storage"
    elif gas_cheaper_pct > 60:
        best = "Heavy gas dispatch + battery for spikes"
    else:
        best = "Battery arbitrage + workload flexibility"
    print(f"    → Best strategy: {best}")

# ============================================================
# 5. DEMAND RESPONSE / GRID SERVICES VALUE
# ============================================================
print("\n[5] Grid Services Revenue Opportunity...")
print("  Can the DC earn money by being flexible?")

# ERCOT demand response: paid to reduce load during emergencies
# Typical: $500-$9000/MWh during scarcity events
# If DC can shed 2 MW during scarcity → gets paid

scarcity_hours = grid_price > 1.0  # >$1000/MWh
n_scarcity = scarcity_hours.sum()
avg_scarcity_price = grid_price[scarcity_hours].mean() * 1000 if n_scarcity > 0 else 0

# If DC can curtail 2MW during scarcity events
DR_CAPACITY = 2000  # kW curtailable
dr_revenue = n_scarcity * DR_CAPACITY * avg_scarcity_price / (1000 * years) if n_scarcity > 0 else 0

print(f"  Scarcity events (>$1000/MWh): {n_scarcity} hours in {years:.0f} years")
print(f"  Avg scarcity price: ${avg_scarcity_price:.0f}/MWh")
print(f"  If DC offers 2MW demand response:")
print(f"    Annual revenue: ${dr_revenue:,.0f}/yr")
print(f"    This is PURE REVENUE — paid to NOT consume!")

# Battery as grid service (frequency regulation)
# ERCOT pays ~$15/MW/hour for regulation capacity
REG_PRICE = 15  # $/MW/hr
# Battery can provide regulation during hours when not needed for arbitrage
# Assume available 30% of hours
reg_hours = len(grid_price) * 0.30
reg_capacity_mw = BATTERY_CAP / 1000 * 0.5  # 50% of battery for regulation
reg_revenue = reg_hours * reg_capacity_mw * REG_PRICE / years

print(f"\n  Battery frequency regulation:")
print(f"    Available capacity: {reg_capacity_mw:.0f} MW for 30% of hours")
print(f"    Annual revenue: ${reg_revenue:,.0f}/yr")

total_grid_services = dr_revenue + reg_revenue
print(f"\n  TOTAL GRID SERVICES REVENUE: ${total_grid_services:,.0f}/yr")
print(f"  → This is money IN, not just savings. Reduces effective operating cost further.")

results["grid_services"] = {
    "demand_response_revenue": float(dr_revenue),
    "regulation_revenue": float(reg_revenue),
    "total_grid_services": float(total_grid_services),
    "scarcity_hours": int(n_scarcity),
}

# ============================================================
# 6. REGIONAL PORTFOLIO DIVERSIFICATION
# ============================================================
print("\n[6] Regional Portfolio Diversification...")
print("  Does operating in multiple regions reduce risk?")

# Merge ERCOT and CAISO prices where they overlap
overlap = ercot.merge(caiso[["timestamp", "lmp_price_usd_mwh"]], on="timestamp",
                       how="inner", suffixes=("_ercot", "_caiso"))

if len(overlap) > 1000:
    # Volatility of single region vs portfolio
    ercot_vol = overlap["lmp_price_usd_mwh_ercot"].std()
    caiso_vol = overlap["lmp_price_usd_mwh_caiso"].std()
    
    # Equal-weight portfolio volatility
    portfolio_price = (overlap["lmp_price_usd_mwh_ercot"] + overlap["lmp_price_usd_mwh_caiso"]) / 2
    portfolio_vol = portfolio_price.std()
    
    # Diversification benefit
    corr = overlap["lmp_price_usd_mwh_ercot"].corr(overlap["lmp_price_usd_mwh_caiso"])
    expected_portfolio_vol = np.sqrt(0.25 * ercot_vol**2 + 0.25 * caiso_vol**2 + 
                                      2 * 0.25 * corr * ercot_vol * caiso_vol)
    diversification_benefit = (1 - portfolio_vol / ((ercot_vol + caiso_vol) / 2)) * 100
    
    print(f"  ERCOT volatility: ${ercot_vol:.1f}/MWh")
    print(f"  CAISO volatility: ${caiso_vol:.1f}/MWh")
    print(f"  Portfolio volatility: ${portfolio_vol:.1f}/MWh")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Diversification benefit: {diversification_benefit:.1f}% volatility reduction")
    
    # Extreme event protection
    ercot_spikes = (overlap["lmp_price_usd_mwh_ercot"] > 500).mean() * 100
    both_spike = ((overlap["lmp_price_usd_mwh_ercot"] > 500) & 
                  (overlap["lmp_price_usd_mwh_caiso"] > 500)).mean() * 100
    
    print(f"\n  Extreme event protection:")
    print(f"    ERCOT spikes >$500: {ercot_spikes:.2f}% of hours")
    print(f"    BOTH regions spike simultaneously: {both_spike:.2f}% of hours")
    print(f"    → Diversification eliminates {(ercot_spikes - both_spike)/ercot_spikes*100:.0f}% of spike exposure")
    
    results["diversification"] = {
        "ercot_volatility": float(ercot_vol),
        "caiso_volatility": float(caiso_vol),
        "portfolio_volatility": float(portfolio_vol),
        "correlation": float(corr),
        "diversification_benefit_pct": float(diversification_benefit),
        "single_region_spike_pct": float(ercot_spikes),
        "both_spike_pct": float(both_spike),
    }

# ============================================================
# 7. TOTAL VALUE STACK (everything combined)
# ============================================================
print("\n" + "=" * 70)
print("[7] TOTAL VALUE STACK — Everything Optena delivers")
print("=" * 70)

# From all our EDAs, compile the complete value proposition
baseline_annual = 4391254  # From EDA 12

value_stack = {
    "Multi-source orchestration (vs grid-only)": 4391254 - 2237890,  # EDA 12
    "Coordination premium (vs isolated rules)": 230649,  # EDA 12
    "Workload flexibility (30% deferral)": 508571,  # EDA 11
    "Grid services revenue": total_grid_services,  # This EDA
    "Cross-regional arbitrage (2 regions)": 145807,  # EDA 10
}

print(f"\n  Baseline (grid-only, 10MW DC): ${baseline_annual:,.0f}/yr\n")
print(f"  {'Value Layer':<45} | {'Annual Value':>12}")
print(f"  {'-'*45} | {'-'*12}")

total_value = 0
for layer, value in value_stack.items():
    print(f"  {layer:<45} | ${value:>10,.0f}")
    total_value += value

print(f"  {'─'*45} | {'─'*12}")
print(f"  {'TOTAL OPTENA VALUE (10MW DC)':<45} | ${total_value:>10,.0f}")
print(f"\n  At 100MW campus (10 DCs): ${total_value * 10:,.0f}/yr")
print(f"  At 500MW fleet (50 DCs): ${total_value * 50:,.0f}/yr")

results["total_value_stack"] = {
    "per_10mw_dc": {k: float(v) for k, v in value_stack.items()},
    "total_per_10mw": float(total_value),
    "total_100mw_campus": float(total_value * 10),
    "total_500mw_fleet": float(total_value * 50),
}

# ============================================================
# SAVE
# ============================================================
print("\n" + "=" * 70)
outpath = os.path.join(RESULTS_DIR, "eda_hidden_value_pareto_results.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  ✓ Saved: {outpath}")
print("=" * 70)
