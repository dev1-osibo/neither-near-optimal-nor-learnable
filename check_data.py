"""Show details of all collected data."""
import pandas as pd
import os
import glob

DATA_DIR = 'patent1-energy-orchestration/data'

print("=" * 70)
print("WEATHER DATA — Open-Meteo (REAL, hourly, 2 years)")
print("=" * 70)
for loc in ['ashburn_va', 'phoenix_az', 'the_dalles_or']:
    f = os.path.join(DATA_DIR, f'weather_{loc}_2024-01-01_2025-12-31.csv')
    df = pd.read_csv(f)
    print(f"\n{loc.upper().replace('_', ' ')}:")
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Variables collected:")
    for col in df.columns:
        if col not in ['timestamp', 'location']:
            print(f"    • {col}: {df[col].min():.1f} to {df[col].max():.1f} (mean: {df[col].mean():.1f})")

print("\n\n" + "=" * 70)
print("EIA DATA — US Government Hourly Grid Monitor (REAL)")
print("=" * 70)
eia_files = sorted(glob.glob(os.path.join(DATA_DIR, 'eia_*.csv')))
for f in eia_files:
    df = pd.read_csv(f)
    name = os.path.basename(f)
    print(f"\n{name}:")
    print(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")
    if 'period' in df.columns:
        print(f"  Period: {df['period'].iloc[-1]} to {df['period'].iloc[0]}")
    if 'type' in df.columns:
        print(f"  Types: {df['type'].unique().tolist()}")
    if 'fueltype' in df.columns:
        print(f"  Fuel types: {df['fueltype'].unique().tolist()}")
    if 'value' in df.columns:
        print(f"  Value range: {df['value'].min()} to {df['value'].max()}")

print("\n\n" + "=" * 70)
print("WHAT'S MISSING")
print("=" * 70)
print("  ❌ Real energy PRICING by region (wholesale $/MWh) — EIA has this")
print("  ❌ Grid carbon intensity — ElectricityMaps or computed from fuel mix")
print("  ❌ CISO fuel type data (timed out) — need to re-fetch")
print("  ❌ Internal DC telemetry (will generate calibrated synthetic)")
