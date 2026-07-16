"""
Fetch REAL Locational Marginal Prices (LMP) from ISO APIs
==========================================================
Sources (VERIFIED WORKING):
  - CAISO OASIS API — 5-min interval LMP, no API key needed ✅
  - ERCOT public data — DAM Settlement Point Prices ✅
  - PJM — requires free API key (user must register at apiportal.pjm.com)

This fetches REAL wholesale electricity market prices.
These are actual settlement prices paid by market participants.

Output: CSV files with columns [timestamp, lmp_price_usd_mwh, region]
"""

import requests
import zipfile
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
from io import BytesIO, StringIO

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# CAISO — California ISO (OASIS API) — VERIFIED WORKING
# Node: TH_NP15_GEN-APND (NP15 trading hub, Northern California)
# This is where The Dalles, OR area data centers are served
# =============================================================================

def fetch_caiso_lmp(start_date: str, end_date: str, 
                    node: str = "TH_NP15_GEN-APND") -> pd.DataFrame:
    """
    Fetch REAL real-time interval LMP from CAISO OASIS API.
    
    Returns 5-minute interval prices, resampled to hourly.
    CAISO limits queries to 31 days at a time.
    Rate limit: ~5 seconds between requests to avoid 429.
    """
    print(f"\n{'='*60}")
    print(f"CAISO OASIS: Fetching real-time LMP for {node}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    all_data = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - current).days
    fetched_days = 0
    
    while current < end:
        # CAISO actually limits to ~7 days for interval data
        chunk_end = min(current + timedelta(days=7), end)
        
        url = "http://oasis.caiso.com/oasisapi/SingleZip"
        params = {
            "queryname": "PRC_INTVL_LMP",
            "market_run_id": "RTM",
            "startdatetime": current.strftime("%Y%m%dT00:00-0000"),
            "enddatetime": chunk_end.strftime("%Y%m%dT00:00-0000"),
            "version": "3",
            "node": node,
            "resultformat": "6",  # CSV
        }
        
        try:
            resp = requests.get(url, params=params, timeout=120)
            
            if resp.status_code == 200 and len(resp.content) > 500:
                z = zipfile.ZipFile(BytesIO(resp.content))
                for fname in z.namelist():
                    if fname.endswith(".csv"):
                        csv_data = z.read(fname).decode("utf-8")
                        df_chunk = pd.read_csv(StringIO(csv_data))
                        if not df_chunk.empty:
                            all_data.append(df_chunk)
                            fetched_days += (chunk_end - current).days
                            print(f"  [{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}] "
                                  f"✓ {len(df_chunk)} rows | Progress: {fetched_days}/{total_days} days")
                    elif fname.endswith(".xml"):
                        # Check for error in XML
                        content = z.read(fname).decode("utf-8")
                        if "ERR_CODE" in content:
                            print(f"  [{current.strftime('%Y-%m-%d')}] API returned error (no data for period)")
                        
            elif resp.status_code == 429:
                print(f"  [{current.strftime('%Y-%m-%d')}] Rate limited — waiting 30s...")
                time.sleep(30)
                continue  # Retry same chunk
            else:
                print(f"  [{current.strftime('%Y-%m-%d')}] HTTP {resp.status_code} ({len(resp.content)} bytes)")
                
        except requests.exceptions.Timeout:
            print(f"  [{current.strftime('%Y-%m-%d')}] Timeout — retrying in 10s...")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"  [{current.strftime('%Y-%m-%d')}] Error: {e}")
        
        current = chunk_end
        time.sleep(6)  # Respect rate limits
    
    if not all_data:
        print("[CAISO] No data retrieved!")
        return pd.DataFrame()
    
    # Combine all chunks
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n  Raw data: {len(df)} rows, columns: {df.columns.tolist()}")
    
    # Filter to LMP only (not congestion/loss components)
    # LMP_TYPE: LMP = total, MCC = congestion, MCL = losses, MCE = energy
    df_lmp = df[df['LMP_TYPE'] == 'LMP'].copy()
    print(f"  After filtering LMP_TYPE=='LMP': {len(df_lmp)} rows")
    
    # Parse timestamps and values
    df_lmp['timestamp'] = pd.to_datetime(df_lmp['INTERVALSTARTTIME_GMT'])
    df_lmp['lmp_price_usd_mwh'] = pd.to_numeric(df_lmp['VALUE'], errors='coerce')
    
    # Keep only what we need
    result = df_lmp[['timestamp', 'lmp_price_usd_mwh']].dropna().copy()
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    # Resample to hourly (average of 5-min intervals within each hour)
    result = result.set_index('timestamp').resample('h').agg(
        lmp_price_usd_mwh=('lmp_price_usd_mwh', 'mean')
    ).reset_index()
    result['region'] = 'CAISO'
    
    print(f"  Final hourly data: {len(result)} rows")
    print(f"  Price range: ${result['lmp_price_usd_mwh'].min():.2f} — ${result['lmp_price_usd_mwh'].max():.2f} /MWh")
    print(f"  Mean price: ${result['lmp_price_usd_mwh'].mean():.2f} /MWh")
    
    return result


# =============================================================================
# ERCOT — via public mis download
# Settlement Point: HB_HOUSTON (Houston Hub)
# =============================================================================

def fetch_ercot_spp(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch ERCOT Day-Ahead Market Settlement Point Prices.
    
    ERCOT publishes historical prices as annual ZIP files.
    Report ID 13060 = Historical DAM Load Zone and Hub Prices
    """
    print(f"\n{'='*60}")
    print(f"ERCOT: Fetching DAM Settlement Point Prices")
    print(f"Period: {start_year} to {end_year}")
    print(f"{'='*60}")
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        # Get document list for this year
        url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
        params = {
            "reportTypeId": "13060",
            "controlsSearch": str(year),
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  [{year}] Doc list HTTP {resp.status_code}")
                continue
                
            data = resp.json()
            doc_list = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
            
            if not doc_list:
                print(f"  [{year}] No documents found")
                continue
            
            # Find the document for this specific year
            target_doc = None
            for doc_entry in doc_list:
                doc = doc_entry.get("Document", {})
                fname = doc.get("FriendlyName", "")
                if str(year) in fname:
                    target_doc = doc
                    break
            
            if not target_doc:
                # Use the first document
                target_doc = doc_list[0].get("Document", doc_list[0])
            
            doc_id = target_doc["DocID"]
            friendly = target_doc.get("FriendlyName", "unknown")
            print(f"  [{year}] Found: {friendly} (DocID: {doc_id})")
            
            # Download the file
            download_url = (
                f"https://www.ercot.com/misdownload/servlets/mirDownload"
                f"?dession=&miession=&def=true&docid={doc_id}"
            )
            resp2 = requests.get(download_url, timeout=120, allow_redirects=True)
            
            if resp2.status_code == 200 and len(resp2.content) > 500:
                try:
                    z = zipfile.ZipFile(BytesIO(resp2.content))
                    for zname in z.namelist():
                        if zname.endswith(".csv"):
                            csv_content = z.read(zname).decode("utf-8")
                            df_chunk = pd.read_csv(StringIO(csv_content))
                            all_data.append(df_chunk)
                            print(f"  [{year}] ✓ {len(df_chunk)} rows from {zname}")
                            break
                except zipfile.BadZipFile:
                    print(f"  [{year}] Downloaded file is not a valid ZIP ({len(resp2.content)} bytes)")
                    # It might be a direct CSV
                    try:
                        df_chunk = pd.read_csv(StringIO(resp2.text))
                        all_data.append(df_chunk)
                        print(f"  [{year}] ✓ Direct CSV: {len(df_chunk)} rows")
                    except:
                        print(f"  [{year}] Could not parse response")
            else:
                print(f"  [{year}] Download failed: HTTP {resp2.status_code}, {len(resp2.content)} bytes")
                
        except Exception as e:
            print(f"  [{year}] Error: {e}")
        
        time.sleep(3)
    
    if not all_data:
        print("[ERCOT] No data retrieved!")
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n  Raw data: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    # ERCOT format typically has: DeliveryDate, HourEnding, SettlementPoint, SettlementPointPrice
    # Standardize
    date_col = [c for c in df.columns if 'date' in c.lower() or 'delivery' in c.lower()]
    hour_col = [c for c in df.columns if 'hour' in c.lower()]
    price_col = [c for c in df.columns if 'price' in c.lower() or 'spp' in c.lower()]
    hub_col = [c for c in df.columns if 'point' in c.lower() or 'hub' in c.lower()]
    
    if date_col and price_col:
        # Filter to Houston hub if possible
        if hub_col:
            hub_values = df[hub_col[0]].unique()
            print(f"  Available hubs: {hub_values[:10]}")
            # Look for HB_HOUSTON or similar
            houston = [h for h in hub_values if 'HOUSTON' in str(h).upper() or 'HB_H' in str(h).upper()]
            if houston:
                df = df[df[hub_col[0]] == houston[0]].copy()
                print(f"  Filtered to {houston[0]}: {len(df)} rows")
        
        # Build timestamp
        if hour_col:
            df['timestamp'] = pd.to_datetime(df[date_col[0]].astype(str)) + \
                              pd.to_timedelta(df[hour_col[0]].astype(int) - 1, unit='h')
        else:
            df['timestamp'] = pd.to_datetime(df[date_col[0]])
        
        result = pd.DataFrame({
            'timestamp': df['timestamp'],
            'lmp_price_usd_mwh': pd.to_numeric(df[price_col[0]], errors='coerce'),
            'region': 'ERCOT'
        })
        result = result.dropna().sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Final: {len(result)} hourly prices")
        if not result.empty:
            print(f"  Price range: ${result['lmp_price_usd_mwh'].min():.2f} — ${result['lmp_price_usd_mwh'].max():.2f} /MWh")
        
        return result
    
    print(f"[ERCOT] Could not identify columns. Available: {df.columns.tolist()}")
    return pd.DataFrame()


# =============================================================================
# PJM — Requires free API key registration
# =============================================================================

def fetch_pjm_lmp(start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch PJM real-time hourly LMP.
    
    Requires a free API key from: https://apiportal.pjm.com/signup/
    If no key provided, prints instructions for user.
    """
    print(f"\n{'='*60}")
    print(f"PJM: Fetching real-time hourly LMP")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    if not api_key:
        print("""
  ⚠️  PJM requires a free API key.
  
  To get one (takes 2 minutes):
  1. Go to: https://apiportal.pjm.com/signup/
  2. Create a free account (no payment needed)
  3. After login, go to Products → subscribe to "Data Miner 2"
  4. Your API key will appear under your Profile
  
  Then re-run this script with:
    python fetch_real_lmp_prices.py --pjm-key YOUR_KEY_HERE
  
  Skipping PJM for now...
""")
        return pd.DataFrame()
    
    all_data = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current < end:
        chunk_end = min(current + timedelta(days=60), end)
        
        url = "https://api.pjm.com/api/v1/rt_hrl_lmps"
        params = {
            "startRow": "1",
            "rowCount": "50000",
            "datetime_beginning_ept": current.strftime("%m/%d/%Y %H:%M"),
            "datetime_ending_ept": chunk_end.strftime("%m/%d/%Y %H:%M"),
            "pnode_id": "51217",  # DOMINION hub (Ashburn, VA)
        }
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "items" in data:
                    df_chunk = pd.DataFrame(data["items"])
                    all_data.append(df_chunk)
                    print(f"  [{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}] ✓ {len(df_chunk)} rows")
            elif resp.status_code == 401:
                print(f"  Invalid API key. Please check your key.")
                return pd.DataFrame()
            else:
                print(f"  [{current.strftime('%Y-%m-%d')}] HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [{current.strftime('%Y-%m-%d')}] Error: {e}")
        
        current = chunk_end
        time.sleep(3)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    result = pd.DataFrame({
        'timestamp': pd.to_datetime(df['datetime_beginning_ept']),
        'lmp_price_usd_mwh': pd.to_numeric(df['total_lmp_rt'], errors='coerce'),
        'region': 'PJM'
    })
    return result.dropna().sort_values('timestamp').reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Fetch all available real LMP prices."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Fetch real LMP electricity prices")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--pjm-key", default=None, help="PJM API key (free from apiportal.pjm.com)")
    parser.add_argument("--region", default="all", choices=["all", "caiso", "ercot", "pjm"],
                        help="Which region to fetch")
    args = parser.parse_args()
    
    results = {}
    
    # --- CAISO ---
    if args.region in ("all", "caiso"):
        caiso_df = fetch_caiso_lmp(args.start, args.end)
        if not caiso_df.empty:
            outpath = os.path.join(DATA_DIR, "real_lmp_CAISO_2020_2025.csv")
            caiso_df.to_csv(outpath, index=False)
            print(f"\n  ✓ Saved CAISO: {outpath} ({len(caiso_df)} rows)")
            results['CAISO'] = caiso_df
    
    # --- ERCOT ---
    if args.region in ("all", "ercot"):
        start_year = int(args.start[:4])
        end_year = int(args.end[:4])
        ercot_df = fetch_ercot_spp(start_year, end_year)
        if not ercot_df.empty:
            outpath = os.path.join(DATA_DIR, "real_lmp_ERCOT_2020_2025.csv")
            ercot_df.to_csv(outpath, index=False)
            print(f"\n  ✓ Saved ERCOT: {outpath} ({len(ercot_df)} rows)")
            results['ERCOT'] = ercot_df
    
    # --- PJM ---
    if args.region in ("all", "pjm"):
        pjm_df = fetch_pjm_lmp(args.start, args.end, api_key=args.pjm_key)
        if not pjm_df.empty:
            outpath = os.path.join(DATA_DIR, "real_lmp_PJM_2020_2025.csv")
            pjm_df.to_csv(outpath, index=False)
            print(f"\n  ✓ Saved PJM: {outpath} ({len(pjm_df)} rows)")
            results['PJM'] = pjm_df
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — REAL LMP PRICES")
    print(f"{'='*60}")
    for name, df in results.items():
        print(f"  {name}: {len(df):,} hourly prices | "
              f"${df['lmp_price_usd_mwh'].mean():.2f}/MWh avg | "
              f"${df['lmp_price_usd_mwh'].min():.2f}–${df['lmp_price_usd_mwh'].max():.2f} range")
    
    if not results:
        print("  No data fetched. Check network and try again.")
    
    return results


if __name__ == "__main__":
    main()
