"""
Run CAISO LMP fetch — 2020-2025 (6 years of real hourly prices).
This takes ~30-45 minutes due to API rate limiting (6s between requests).

Saves progress incrementally so it can be resumed if interrupted.
"""

import sys
import os
import time
import requests
import zipfile
import pandas as pd
from datetime import timedelta
from io import BytesIO, StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "real_lmp_CAISO_2020_2025.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "_caiso_fetch_progress.csv")
NODE = "TH_NP15_GEN-APND"
START = "2023-06-01"
END = "2025-12-31"


def fetch_chunk(start_dt, end_dt, node):
    """Fetch one 7-day chunk from CAISO OASIS."""
    url = "http://oasis.caiso.com/oasisapi/SingleZip"
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "startdatetime": start_dt.strftime("%Y%m%dT00:00-0000"),
        "enddatetime": end_dt.strftime("%Y%m%dT00:00-0000"),
        "version": "3",
        "node": node,
        "resultformat": "6",
    }
    
    resp = requests.get(url, params=params, timeout=120)
    
    if resp.status_code == 429:
        return "rate_limited", None
    
    if resp.status_code != 200 or len(resp.content) < 500:
        return "no_data", None
    
    try:
        z = zipfile.ZipFile(BytesIO(resp.content))
        for fname in z.namelist():
            if fname.endswith(".csv"):
                csv_data = z.read(fname).decode("utf-8")
                df = pd.read_csv(StringIO(csv_data))
                # Filter to LMP only
                df_lmp = df[df["LMP_TYPE"] == "LMP"].copy()
                if df_lmp.empty:
                    return "no_lmp", None
                
                df_lmp["timestamp"] = pd.to_datetime(df_lmp["INTERVALSTARTTIME_GMT"])
                df_lmp["lmp_price_usd_mwh"] = pd.to_numeric(df_lmp["VALUE"], errors="coerce")
                result = df_lmp[["timestamp", "lmp_price_usd_mwh"]].dropna()
                return "success", result
            elif fname.endswith(".xml"):
                content = z.read(fname).decode("utf-8")
                if "ERR_CODE" in content:
                    return "api_error", None
    except zipfile.BadZipFile:
        return "bad_zip", None
    
    return "unknown", None


def main():
    print(f"{'='*60}")
    print(f"CAISO REAL LMP FETCH — {START} to {END}")
    print(f"Node: {NODE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    
    # Check for existing progress
    all_data = []
    last_fetched = pd.Timestamp(START)
    
    if os.path.exists(PROGRESS_FILE):
        existing = pd.read_csv(PROGRESS_FILE)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        all_data.append(existing)
        last_fetched = existing["timestamp"].max() - timedelta(hours=1)
        print(f"Resuming from {last_fetched.strftime('%Y-%m-%d')} ({len(existing)} existing rows)\n")
    
    current = last_fetched if last_fetched > pd.Timestamp(START) else pd.Timestamp(START)
    end = pd.Timestamp(END)
    total_weeks = int((end - current).days / 7) + 1
    week_num = 0
    consecutive_errors = 0
    
    while current < end:
        chunk_end = min(current + timedelta(days=7), end)
        week_num += 1
        
        status, data = fetch_chunk(current, chunk_end, NODE)
        
        if status == "success" and data is not None:
            all_data.append(data)
            consecutive_errors = 0
            rows = len(data)
            pct = (week_num / total_weeks) * 100
            print(f"  [{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}] "
                  f"✓ {rows} intervals | Week {week_num}/{total_weeks} ({pct:.0f}%)")
            
            # Save progress every 10 chunks
            if week_num % 10 == 0:
                progress_df = pd.concat(all_data, ignore_index=True)
                progress_df.to_csv(PROGRESS_FILE, index=False)
                print(f"    → Progress saved: {len(progress_df)} total rows")
            
        elif status == "rate_limited":
            print(f"  [{current.strftime('%Y-%m-%d')}] Rate limited — waiting 60s...")
            time.sleep(60)
            continue  # Retry
            
        elif status == "api_error":
            consecutive_errors += 1
            print(f"  [{current.strftime('%Y-%m-%d')}] No data for period (error #{consecutive_errors})")
            if consecutive_errors >= 5:
                print(f"  ⚠️  5 consecutive errors — data may not exist for this period")
                consecutive_errors = 0
        else:
            consecutive_errors += 1
            print(f"  [{current.strftime('%Y-%m-%d')}] {status} (error #{consecutive_errors})")
        
        current = chunk_end
        time.sleep(6)  # Rate limit compliance
    
    # Final assembly
    if not all_data:
        print("\n❌ No data fetched!")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # Resample to hourly
    df_hourly = df.set_index("timestamp").resample("h").agg(
        lmp_price_usd_mwh=("lmp_price_usd_mwh", "mean")
    ).reset_index()
    df_hourly["region"] = "CAISO"
    df_hourly = df_hourly.dropna()
    
    # Save final
    df_hourly.to_csv(OUTPUT_FILE, index=False)
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"  Hourly prices: {len(df_hourly)}")
    print(f"  Date range: {df_hourly['timestamp'].min()} → {df_hourly['timestamp'].max()}")
    print(f"  Price range: ${df_hourly['lmp_price_usd_mwh'].min():.2f} → ${df_hourly['lmp_price_usd_mwh'].max():.2f} /MWh")
    print(f"  Mean price: ${df_hourly['lmp_price_usd_mwh'].mean():.2f} /MWh")
    print(f"  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
