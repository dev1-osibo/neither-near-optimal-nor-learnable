import json

# TFT vs Prophet at each horizon
complete = json.load(open('results/complete_model_comparison.json'))

print('=== TFT vs Prophet (internal) ===')
for h in ['1h','4h','12h','24h']:
    tft = complete['all_models']['TFT (fusion, 168h lookback)'][h]
    prop = complete['all_models']['Prophet (internal - univariate)'][h]
    impr = (prop - tft)/prop * 100
    print(f'{h}: TFT={tft:.2f}%, Prophet={prop:.2f}%, improvement={impr:.1f}%')

tft_mapes = [complete['all_models']['TFT (fusion, 168h lookback)'][h] for h in ['1h','4h','12h','24h']]
print(f'\nTFT avg MAPE: {sum(tft_mapes)/len(tft_mapes):.2f}%')

# Linear fusion improvement
lin_int_12h = complete['all_models']['Linear (internal lags)']['12h']
lin_fus_12h = complete['all_models']['Linear (fusion lags)']['12h']
tft_12h = complete['all_models']['TFT (fusion, 168h lookback)']['12h']
print(f'\nLinear internal at 12h: {lin_int_12h:.2f}%')
print(f'Linear fusion at 12h: {lin_fus_12h:.2f}%')
print(f'TFT at 12h: {tft_12h:.2f}%')
print(f'Fusion (linear) improvement at 12h: {(lin_int_12h-lin_fus_12h)/lin_int_12h*100:.1f}%')
print(f'Fusion (TFT) improvement over linear internal at 12h: {(lin_int_12h-tft_12h)/lin_int_12h*100:.1f}%')

# From NB02 deep results
deep = json.load(open('results/eda_deep_results.json'))
print(f'\nNB02 progressive model G improvement (linear+lags): {deep["progressive_model_comparison"]["G_plus_lags"]["improvement_pct"]:.1f}%')

pareto = json.load(open('results/eda_hidden_value_pareto_results.json'))
bc = pareto['pareto_frontier']['baseline_cost']
bcarbon = pareto['pareto_frontier']['baseline_carbon']
pt_01 = [p for p in pareto['pareto_frontier']['points'] if abs(p['alpha']-0.1)<0.01][0]
cost_save = (bc - pt_01['cost'])/bc*100
carbon_save = (bcarbon - pt_01['carbon'])/bcarbon*100
print(f'\n=== Pareto (alpha=0.1) ===')
print(f'Cost reduction: {cost_save:.1f}%')
print(f'Carbon reduction: {carbon_save:.1f}%')

total_stack = pareto['total_value_stack']['total_per_10mw']
print(f'\n=== Total value stack per 10MW: ${total_stack:,.0f}')

stress = json.load(open('results/eda_stress_testing_results.json'))
print(f'\n=== Texas Freeze savings: ${stress["texas_freeze"]["saving_one_week"]:,.0f}')

coord = json.load(open('results/eda_coordination_value_results.json'))
print(f'\n=== Coordination premium: {coord["coordination_value"]["coordination_premium_vs_isolated_pct"]:.1f}%')
print(f'Coordination value $: ${coord["coordination_value"]["coordination_value_dollars_vs_isolated"]:,.0f}')
zero_capex = coord["coordination_value"]["baseline_annual_cost"] - coord["coordination_value"]["coordinated_annual_cost"]
print(f'Coordinated vs baseline (annual): ${zero_capex:,.0f}')

multi = json.load(open('results/eda_multi_source_energy_results.json'))
print(f'\n=== Wind capacity factor: {multi["wind"]["capacity_factor_pct"]:.1f}%')
print(f'Solar capacity factor: {multi["solar"]["capacity_factor_pct"]:.1f}%')
print(f'Gas cheaper than ERCOT: {multi["gas_analysis"]["pct_hours_gas_cheaper_than_ercot"]:.1f}%')

new_sav = json.load(open('results/eda_new_savings_angles_results.json'))
print(f'\n=== Zero-CAPEX total (NB14): ${new_sav["grid_only_savings"]["total_zero_capex"]:,.0f}')

moat = json.load(open('results/eda_moat_and_scaling_results.json'))
print(f'\n=== $/MW/yr scaling: ${moat["scaling"]["1MW"]["per_mw_saving"]:,.0f}')

# Check CAISO date coverage for cross-regional arbitrage
import pandas as pd
caiso = pd.read_csv('data/real_lmp_CAISO_2020_2025.csv')
ercot = pd.read_csv('data/real_lmp_ERCOT_2020_2025.csv')
print(f'\nERCOT coverage: {ercot["timestamp"].min()} to {ercot["timestamp"].max()} ({len(ercot)} hours)')
print(f'CAISO coverage: {caiso["timestamp"].min()} to {caiso["timestamp"].max()} ({len(caiso)} hours)')
print(f'CAISO only covers {len(caiso)/len(ercot)*100:.0f}% of ERCOT period')
