"""
FULL DATA COLLECTION — Jan 1, 2020 to Dec 25, 2025
All datasets, all regions, no shortcuts.
"""
import requests
import pandas as pd
import os
import time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY = "29E8nLYJS8QZro2mhfIieTrMNSoiinPwClKblFAt"

# ============================================================
# WEATHER — Open-Meteo (free, no key, chunked by year)
# ============================================================

LOCATIONS = {
    'ashburn_va': {'lat': 39.04, 'lon': -77.49, 'tz': 'America/New_York'},
    'phoenix_az': {'lat': 33.45, 'lon': -112.07, 'tz': 'America/Phoenix'},
    'the_dalles_or': {'lat': 45.60, 'lon': -121.18, 'tz': 'America/Los_Angeles'},
}

WEATHER_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
    'surface_pressure', 'cloud_cover', 'wind_speed_10m',
    'direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
]

# Open-Meteo archive API allows up to ~2 year chunks
YEAR_CHUNKS = [
    ('2020-01-01', '2021-12-31'),
    ('2022-01-01', '2023-12-31'),
    ('2024-01-01', '2025-12-25'),
]


def fetch_weather_full():
    """Fetch full 6-year weather data for all 3 locations."""
    print("=" * 70)
    print("WEATHER DATA — Open-Meteo Archive API")
    print("Period: 2020-01-01 to 2025-12-25 (6 years)")
    print("=" * 70)
    
    for loc_key, loc_info in LOCATIONS.items():
        print(f"\n  {loc_key.upper().replace('_', ' ')}:")
        all_dfs = []
        
        for start, end in YEAR_CHUNKS:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': loc_info['lat'],
                'longitude': loc_info['lon'],
                'start_date': start,
                'end_date': end,
                'hourly': ','.join(WEATHER_VARS),
                'timezone': loc_info['tz'],
            }
            
            print(f"    Fetching {start} to {end}...", end=' ')
            try:
                resp = requests.get(url, params=params, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'hourly' in data:
                        hourly = data['hourly']
                        df = pd.DataFrame({
                            'timestamp': pd.to_datetime(hourly['time']),
                            **{v: hourly.get(v) for v in WEATHER_VARS}
                        })
                        all_dfs.append(df)
                        print(f"{len(df):,} rows ✓")
                    else:
                        print("No hourly data")
                else:
                    print(f"HTTP {resp.status_code}")
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(1)  # Rate limiting
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined['location'] = loc_key
            filepath = os.path.join(DATA_DIR, f'weather_{loc_key}_2020_2025.csv')
            combined.to_csv(filepath, index=False)
            print(f"    TOTAL: {len(combined):,} rows saved to {os.path.basename(filepath)}")


# ============================================================
# EIA — All datasets, full period, with pagination
# ============================================================

EIA_BASE = "https://api.eia.gov/v2/electricity/rto"
REGIONS = ['PJM', 'ERCO', 'CISO']

EIA_DATASETS = {
    'demand': {
        'endpoint': '/region-data/data/',
        'facet_key': 'respondent',
    },
    'interchange': {
        'endpoint': '/interchange-data/data/',
        'facet_key': 'fromba',
    },
    'fuel_type': {
        'endpoint': '/fuel-type-data/data/',
        'facet_key': 'respondent',
    },
    'daily_demand': {
        'endpoint': '/daily-region-data/data/',
        'facet_key': 'respondent',
    },
}


def fetch_eia_full(dataset_key, region, start='2020-01-01T00', end='2025-12-25T00'):
    """Fetch full EIA dataset with complete pagination."""
    ds = EIA_DATASETS[dataset_key]
    url = f"{EIA_BASE}{ds['endpoint']}"
    
    all_records = []
    offset = 0
    batch_size = 5000
    max_total = 500000  # Safety limit
    
    while offset < max_total:
        params = {
            'api_key': API_KEY,
            'frequency': 'hourly',
            'data[0]': 'value',
            f"facets[{ds['facet_key']}][]": region,
            'start': start,
            'end': end,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': offset,
            'length': batch_size,
        }
        
        # Daily endpoint uses daily frequency
        if dataset_key == 'daily_demand':
            params['frequency'] = 'daily'
            params['start'] = '2020-01-01'
            params['end'] = '2025-12-25'
        
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                break
            
            data = resp.json()
            if 'response' not in data or 'data' not in data['response']:
                break
            
            records = data['response']['data']
            if not records:
                break
            
            all_records.extend(records)
            offset += batch_size
            
            if len(records) < batch_size:
                break  # Last page
            
            time.sleep(0.3)  # Rate limit
            
        except Exception as e:
            print(f"      Error at offset {offset}: {e}")
            break
    
    return all_records


def fetch_all_eia():
    """Fetch all EIA datasets for all regions, full 6-year period."""
    print("\n\n" + "=" * 70)
    print("EIA DATA — Hourly Electric Grid Monitor")
    print("Period: 2020-01-01 to 2025-12-25 (6 years)")
    print("=" * 70)
    
    for dataset_key in EIA_DATASETS:
        print(f"\n  Dataset: {dataset_key}")
        print(f"  {'─' * 50}")
        
        for region in REGIONS:
            print(f"    {region}: fetching...", end=' ')
            
            records = fetch_eia_full(dataset_key, region)
            
            if records:
                df = pd.DataFrame(records)
                filepath = os.path.join(DATA_DIR, f'eia_{dataset_key}_{region}_full.csv')
                df.to_csv(filepath, index=False)
                print(f"{len(df):,} rows saved ✓")
                
                if 'period' in df.columns:
                    print(f"           Period: {df['period'].iloc[0]} to {df['period'].iloc[-1]}")
            else:
                print("0 rows (no data)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  FULL DATA COLLECTION — Jan 1, 2020 to Dec 25, 2025            ║")
    print("║  Weather (3 locations) + EIA (4 datasets × 3 regions)          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    start_time = time.time()
    
    # Weather
    fetch_weather_full()
    
    # EIA
    fetch_all_eia()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Runtime: {elapsed/60:.1f} minutes")
    
    # Count all files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and ('2020' in f or '_full' in f)]
    total_rows = 0
    for f in sorted(files):
        filepath = os.path.join(DATA_DIR, f)
        rows = sum(1 for _ in open(filepath)) - 1
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        total_rows += rows
        print(f"  {f}: {rows:,} rows ({size_mb:.1f} MB)")
    
    print(f"\nTOTAL: {total_rows:,} rows across {len(files)} files")
