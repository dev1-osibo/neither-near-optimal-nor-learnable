"""Fetch electricity pricing data from EIA."""
import requests
import pandas as pd
import os
import time

API_KEY = os.environ.get("EIA_API_KEY", "")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Retail prices (monthly, by state) — commercial sector
print("Fetching retail electricity prices (monthly by state, commercial sector)...")
print("States: VA, AZ, OR, TX, CA | Period: 2020-01 to 2025-12")

url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
all_records = []

for state in ['VA', 'AZ', 'OR', 'TX', 'CA']:
    params = {
        'api_key': API_KEY,
        'frequency': 'monthly',
        'data[0]': 'price',
        'data[1]': 'revenue',
        'data[2]': 'sales',
        'facets[stateid][]': state,
        'facets[sectorid][]': 'COM',
        'start': '2020-01',
        'end': '2025-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 200:
        data = r.json()
        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
            all_records.extend(records)
            print(f"  {state}: {len(records)} monthly records")
    time.sleep(0.3)

if all_records:
    df = pd.DataFrame(all_records)
    filepath = os.path.join(DATA_DIR, 'eia_retail_prices_full.csv')
    df.to_csv(filepath, index=False)
    print(f"\n  SAVED: {filepath}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    # Show sample
    if 'price' in df.columns:
        print(f"\n  Price summary (cents/kWh):")
        for state in df['stateid'].unique():
            state_df = df[df['stateid'] == state]
            prices = pd.to_numeric(state_df['price'], errors='coerce').dropna()
            if len(prices) > 0:
                print(f"    {state}: min={prices.min():.1f}, max={prices.max():.1f}, mean={prices.mean():.1f} cents/kWh")

# Also get industrial prices (large DC operators get industrial rates)
print("\n\nFetching industrial electricity prices...")
all_ind = []
for state in ['VA', 'AZ', 'OR', 'TX', 'CA']:
    params['facets[sectorid][]'] = 'IND'
    params['facets[stateid][]'] = state
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 200:
        data = r.json()
        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
            all_ind.extend(records)
            print(f"  {state}: {len(records)} rows")
    time.sleep(0.3)

if all_ind:
    df_ind = pd.DataFrame(all_ind)
    filepath = os.path.join(DATA_DIR, 'eia_industrial_prices_full.csv')
    df_ind.to_csv(filepath, index=False)
    print(f"\n  SAVED: {filepath} ({len(df_ind):,} rows)")
    if 'price' in df_ind.columns:
        print(f"  Industrial price summary (cents/kWh):")
        for state in df_ind['stateid'].unique():
            state_df = df_ind[df_ind['stateid'] == state]
            prices = pd.to_numeric(state_df['price'], errors='coerce').dropna()
            if len(prices) > 0:
                print(f"    {state}: min={prices.min():.1f}, max={prices.max():.1f}, mean={prices.mean():.1f} cents/kWh")

print("\nDone.")
