"""
Fetch REAL retail electricity prices from EIA API
==================================================
Source: EIA Form 826/861 — Monthly retail electricity prices by state and sector.
API Key: set via the EIA_API_KEY environment variable

Purpose: Validation/scaling reference for hourly LMP data.
Gives us real average price levels (cents/kWh) for:
  - Virginia (PJM territory — Ashburn data centers)
  - Texas (ERCOT territory)  
  - California (CAISO territory — The Dalles adjacent)
  - Oregon (CAISO/BPA territory — The Dalles)

Output: CSV with monthly prices for industrial and commercial sectors.
"""

import requests
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

# States relevant to our 3 data center regions
TARGET_STATES = {
    "VA": "Virginia (PJM — Ashburn)",
    "TX": "Texas (ERCOT)",
    "CA": "California (CAISO)",
    "OR": "Oregon (The Dalles)",
}

# Sectors: COM = Commercial, IND = Industrial (data centers are typically industrial)
TARGET_SECTORS = ["COM", "IND"]


def fetch_eia_retail_prices(start: str = "2020-01", end: str = "2025-12") -> pd.DataFrame:
    """
    Fetch monthly retail electricity prices from EIA API v2.
    
    Returns price in cents/kWh, revenue in thousand dollars, sales in MWh.
    """
    print(f"\n{'='*60}")
    print(f"EIA API: Fetching monthly retail electricity prices")
    print(f"Period: {start} to {end}")
    print(f"States: {list(TARGET_STATES.keys())}")
    print(f"Sectors: {TARGET_SECTORS}")
    print(f"{'='*60}")
    
    all_data = []
    
    for state_code, state_desc in TARGET_STATES.items():
        for sector in TARGET_SECTORS:
            url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
            params = {
                "api_key": EIA_API_KEY,
                "frequency": "monthly",
                "data[0]": "price",
                "data[1]": "revenue",
                "data[2]": "sales",
                "facets[stateid][]": state_code,
                "facets[sectorid][]": sector,
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": 0,
                "length": 5000,
            }
            
            try:
                resp = requests.get(url, params=params, timeout=60)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "response" in data and "data" in data["response"]:
                        items = data["response"]["data"]
                        if items:
                            df_chunk = pd.DataFrame(items)
                            df_chunk["state_code"] = state_code
                            df_chunk["state_desc"] = state_desc
                            all_data.append(df_chunk)
                            print(f"  ✓ {state_code}/{sector}: {len(items)} months")
                        else:
                            print(f"  - {state_code}/{sector}: no data")
                    else:
                        print(f"  - {state_code}/{sector}: unexpected response format")
                else:
                    print(f"  ✗ {state_code}/{sector}: HTTP {resp.status_code}")
                    
            except Exception as e:
                print(f"  ✗ {state_code}/{sector}: {e}")
    
    if not all_data:
        print("No data retrieved!")
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Clean up
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    df["price_cents_kwh"] = pd.to_numeric(df["price"], errors="coerce")
    df["revenue_thousand_usd"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["sales_mwh"] = pd.to_numeric(df["sales"], errors="coerce")
    
    # Convert price to $/MWh for consistency with LMP data
    df["price_usd_mwh"] = df["price_cents_kwh"] * 10  # cents/kWh → $/MWh
    
    print(f"\n  Total records: {len(df)}")
    print(f"\n  Price summary by state (Industrial, $/MWh):")
    ind = df[df["sectorid"] == "IND"] if "sectorid" in df.columns else df
    for state in TARGET_STATES:
        state_data = ind[ind["state_code"] == state]["price_usd_mwh"]
        if not state_data.empty:
            print(f"    {state}: ${state_data.mean():.1f}/MWh avg "
                  f"(${state_data.min():.1f} – ${state_data.max():.1f})")
    
    return df


def main():
    df = fetch_eia_retail_prices()
    
    if not df.empty:
        outpath = os.path.join(DATA_DIR, "real_eia_retail_prices_2020_2025.csv")
        df.to_csv(outpath, index=False)
        print(f"\n  ✓ Saved: {outpath} ({len(df)} rows)")
        
        # Also save a summary pivot table
        if "sectorid" in df.columns:
            pivot = df.pivot_table(
                index="period", 
                columns=["state_code", "sectorid"],
                values="price_usd_mwh"
            )
            pivot_path = os.path.join(DATA_DIR, "real_eia_retail_prices_pivot.csv")
            pivot.to_csv(pivot_path)
            print(f"  ✓ Pivot: {pivot_path}")
    
    return df


if __name__ == "__main__":
    main()
