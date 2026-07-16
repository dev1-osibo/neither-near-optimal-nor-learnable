"""
Fetch REAL natural gas prices from EIA API
============================================
Source: EIA Henry Hub Natural Gas Spot Price (daily)
+ EIA Natural Gas Electric Power Price (monthly, $/MCF)

These are the actual market prices for natural gas used in power generation.
For on-site gas generators/fuel cells, cost = gas_price / generator_efficiency.

API Key: set via the EIA_API_KEY environment variable
"""

import requests
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")


def fetch_henry_hub_spot():
    """
    Fetch Henry Hub Natural Gas Spot Price (daily).
    Series: NG.RNGWHHD.D (Henry Hub spot price, $/MMBTU)
    """
    print("=" * 60)
    print("EIA: Fetching Henry Hub Natural Gas Spot Price (daily)")
    print("=" * 60)
    
    # EIA APIv2 — Natural Gas spot prices
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RNGWHHD",  # Henry Hub spot
        "start": "2020-01-01",
        "end": "2025-12-31",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    
    resp = requests.get(url, params=params, timeout=60)
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        if "response" in data and "data" in data["response"]:
            items = data["response"]["data"]
            print(f"  Got {len(items)} daily prices")
            if items:
                df = pd.DataFrame(items)
                df["period"] = pd.to_datetime(df["period"])
                df["gas_price_usd_mmbtu"] = pd.to_numeric(df["value"], errors="coerce")
                df = df[["period", "gas_price_usd_mmbtu"]].dropna()
                df = df.rename(columns={"period": "date"})
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Price range: ${df['gas_price_usd_mmbtu'].min():.2f} to ${df['gas_price_usd_mmbtu'].max():.2f} /MMBTU")
                print(f"  Mean: ${df['gas_price_usd_mmbtu'].mean():.2f} /MMBTU")
                return df
        else:
            print(f"  Unexpected response: {list(data.keys())}")
            if "error" in data:
                print(f"  Error: {data['error']}")
    else:
        print(f"  HTTP error: {resp.text[:300]}")
    
    return pd.DataFrame()


def fetch_gas_electric_power_price():
    """
    Fetch Natural Gas Price for Electric Power (monthly, $/MCF).
    This is what power plants actually pay — more relevant than Henry Hub for generation cost.
    """
    print("\n" + "=" * 60)
    print("EIA: Fetching Natural Gas Electric Power Price (monthly)")
    print("=" * 60)
    
    url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "monthly",
        "data[0]": "value",
        "facets[process][]": "PEU",  # Price, Electric Power
        "facets[duoarea][]": "NUS",  # National US
        "start": "2020-01",
        "end": "2025-12",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    
    resp = requests.get(url, params=params, timeout=60)
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        if "response" in data and "data" in data["response"]:
            items = data["response"]["data"]
            print(f"  Got {len(items)} monthly prices")
            if items:
                df = pd.DataFrame(items)
                df["period"] = pd.to_datetime(df["period"])
                df["gas_elec_price_usd_mcf"] = pd.to_numeric(df["value"], errors="coerce")
                df = df[["period", "gas_elec_price_usd_mcf"]].dropna()
                df = df.rename(columns={"period": "date"})
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Price range: ${df['gas_elec_price_usd_mcf'].min():.2f} to ${df['gas_elec_price_usd_mcf'].max():.2f} /MCF")
                print(f"  Mean: ${df['gas_elec_price_usd_mcf'].mean():.2f} /MCF")
                return df
    else:
        print(f"  Response: {resp.text[:300]}")
    
    return pd.DataFrame()


def main():
    # Fetch both price series
    henry_hub = fetch_henry_hub_spot()
    elec_power = fetch_gas_electric_power_price()
    
    # Save Henry Hub daily
    if not henry_hub.empty:
        outpath = os.path.join(DATA_DIR, "real_gas_henry_hub_daily_2020_2025.csv")
        henry_hub.to_csv(outpath, index=False)
        print(f"\n  ✓ Saved: {outpath} ({len(henry_hub)} daily prices)")
        
        # Also convert to $/kWh for direct comparison with electricity
        # 1 MMBTU = 293.07 kWh, gas generator efficiency = 40%
        # Cost per kWh generated = price_per_mmbtu / (293.07 * 0.40)
        henry_hub["gas_cost_usd_kwh"] = henry_hub["gas_price_usd_mmbtu"] / (293.07 * 0.40)
        henry_hub["gas_cost_usd_mwh"] = henry_hub["gas_cost_usd_kwh"] * 1000
        print(f"  Equivalent electricity cost (40% efficiency):")
        print(f"    ${henry_hub['gas_cost_usd_mwh'].mean():.1f}/MWh avg")
        print(f"    ${henry_hub['gas_cost_usd_mwh'].min():.1f} - ${henry_hub['gas_cost_usd_mwh'].max():.1f} range")
    
    # Save electric power price monthly
    if not elec_power.empty:
        outpath = os.path.join(DATA_DIR, "real_gas_electric_power_monthly_2020_2025.csv")
        elec_power.to_csv(outpath, index=False)
        print(f"\n  ✓ Saved: {outpath} ({len(elec_power)} monthly prices)")
    
    print("\n" + "=" * 60)
    print("GAS DATA SUMMARY")
    print("=" * 60)
    print("  On-site gas generator assumptions:")
    print("    - Type: Natural gas reciprocating engine or fuel cell")
    print("    - Capacity: 2 MW (typical for large DC backup/peaker)")
    print("    - Efficiency: 40% (reciprocating) or 60% (fuel cell)")
    print("    - Emissions: 0.41 kg CO2/kWh (gas) vs grid average ~0.37")
    print("    - Use case: Dispatchable peaker during high grid prices")
    print("    - Decision: Run gas when grid_price > gas_cost AND carbon budget allows")


if __name__ == "__main__":
    main()
