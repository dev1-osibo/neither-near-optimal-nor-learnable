"""
Fetch all EIA Hourly Electric Grid Monitor data.
4 datasets × 3 regions = 12 API calls.
"""
import requests
import pandas as pd
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY = "29E8nLYJS8QZro2mhfIieTrMNSoiinPwClKblFAt"
BASE_URL = "https://api.eia.gov/v2/electricity/rto"

REGIONS = ['PJM', 'ERCO', 'CISO']
REGION_NAMES = {'PJM': 'Virginia/DC', 'ERCO': 'Texas', 'CISO': 'California'}

DATASETS = {
    'demand': {
        'endpoint': '/region-data/data/',
        'description': 'Hourly demand, day-ahead forecast, net generation',
        'facet_key': 'respondent',
    },
    'interchange': {
        'endpoint': '/interchange-data/data/',
        'description': 'Hourly interchange between balancing authorities',
        'facet_key': 'fromba',
    },
    'fuel_type': {
        'endpoint': '/fuel-type-data/data/',
        'description': 'Hourly generation by energy source (solar, wind, gas, nuclear)',
        'facet_key': 'respondent',
    },
}


def fetch_eia(dataset_key, region, start='2024-01-01T00', end='2025-12-31T00', max_records=50000):
    """Fetch data from EIA API with pagination."""
    ds = DATASETS[dataset_key]
    url = f"{BASE_URL}{ds['endpoint']}"
    
    all_records = []
    offset = 0
    batch_size = 5000
    
    while offset < max_records:
        params = {
            'api_key': API_KEY,
            'frequency': 'hourly',
            'data[0]': 'value',
            f'facets[{ds["facet_key"]}][]': region,
            'start': start,
            'end': end,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': offset,
            'length': batch_size,
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != 200:
                print(f"    HTTP {response.status_code}: {response.text[:100]}")
                break
            
            data = response.json()
            if 'response' not in data or 'data' not in data['response']:
                print(f"    No data in response at offset {offset}")
                break
            
            records = data['response']['data']
            if not records:
                break
            
            all_records.extend(records)
            offset += batch_size
            
            # Rate limiting
            time.sleep(0.5)
            
            if len(records) < batch_size:
                break  # Last page
                
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    return all_records


def main():
    print("=" * 70)
    print("EIA DATA COLLECTION — Hourly Electric Grid Monitor")
    print(f"API Key: {API_KEY[:8]}...")
    print(f"Regions: {', '.join(f'{r} ({REGION_NAMES[r]})' for r in REGIONS)}")
    print(f"Period: 2024-01-01 to 2025-12-31")
    print("=" * 70)
    
    total_rows = 0
    files_saved = []
    
    for dataset_key, ds_info in DATASETS.items():
        print(f"\n{'─'*50}")
        print(f"Dataset: {dataset_key} — {ds_info['description']}")
        print(f"{'─'*50}")
        
        for region in REGIONS:
            print(f"\n  Fetching {dataset_key} for {region} ({REGION_NAMES[region]})...")
            
            records = fetch_eia(dataset_key, region)
            
            if records:
                df = pd.DataFrame(records)
                filename = f"eia_{dataset_key}_{region}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                df.to_csv(filepath, index=False)
                
                total_rows += len(df)
                files_saved.append(filename)
                
                print(f"    Saved: {filename} ({len(df):,} rows, {len(df.columns)} cols)")
                if 'period' in df.columns:
                    print(f"    Period: {df['period'].min()} to {df['period'].max()}")
                if 'value' in df.columns:
                    print(f"    Value range: {df['value'].min()} to {df['value'].max()}")
            else:
                print(f"    No data returned for {region}")
    
    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total rows collected: {total_rows:,}")
    print(f"Files saved: {len(files_saved)}")
    for f in files_saved:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
