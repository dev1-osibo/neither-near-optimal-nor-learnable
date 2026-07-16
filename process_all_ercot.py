"""
Process ALL ERCOT DAM Settlement Point Price ZIP files from datasets folder.
These are nested ZIPs (outer zip contains daily zips, each with a CSV).
Extract HB_HOUSTON hourly prices for full date range.
"""

import zipfile
import pandas as pd
import os
from io import BytesIO

DATASETS_DIR = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\datasets"
OUTPUT_DIR = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\data"

# Find all ERCOT NP4-190 ZIP files
ercot_zips = sorted([
    f for f in os.listdir(DATASETS_DIR) 
    if f.startswith("cdr.np4-190") and f.endswith(".zip")
])

print(f"Found {len(ercot_zips)} ERCOT ZIP files in datasets/")
for zf in ercot_zips:
    size_mb = os.path.getsize(os.path.join(DATASETS_DIR, zf)) / (1024*1024)
    print(f"  {zf} ({size_mb:.1f} MB)")

# Process all files
all_houston = []
total_inner_zips = 0
errors = 0

for zf in ercot_zips:
    path = os.path.join(DATASETS_DIR, zf)
    print(f"\nProcessing: {zf}")
    
    try:
        outer = zipfile.ZipFile(path)
        inner_names = outer.namelist()
        print(f"  Contains {len(inner_names)} inner files")
        
        for inner_name in inner_names:
            try:
                # Each inner file is also a ZIP containing a CSV
                inner_content = outer.read(inner_name)
                
                if inner_name.endswith(".zip"):
                    inner_zip = zipfile.ZipFile(BytesIO(inner_content))
                    for csv_name in inner_zip.namelist():
                        if csv_name.endswith(".csv"):
                            df = pd.read_csv(BytesIO(inner_zip.read(csv_name)))
                            # Filter to HB_HOUSTON
                            hb = df[df["SettlementPoint"] == "HB_HOUSTON"].copy()
                            if not hb.empty:
                                all_houston.append(hb)
                            total_inner_zips += 1
                elif inner_name.endswith(".csv"):
                    df = pd.read_csv(BytesIO(inner_content))
                    hb = df[df["SettlementPoint"] == "HB_HOUSTON"].copy()
                    if not hb.empty:
                        all_houston.append(hb)
                    total_inner_zips += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error on {inner_name}: {e}")
                    
    except Exception as e:
        print(f"  ERROR opening {zf}: {e}")

print(f"\n{'='*60}")
print(f"Processed {total_inner_zips} daily files, {errors} errors")

if not all_houston:
    print("NO DATA EXTRACTED!")
else:
    df = pd.concat(all_houston, ignore_index=True)
    print(f"Raw HB_HOUSTON rows: {len(df):,}")
    
    # Build timestamp
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"])
    # HourEnding format: "01:00", "02:00", ..., "24:00"
    df["hour"] = df["HourEnding"].astype(str).str.replace(":00", "").str.strip().astype(int) - 1
    df["timestamp"] = df["DeliveryDate"] + pd.to_timedelta(df["hour"], unit="h")
    df["lmp_price_usd_mwh"] = pd.to_numeric(df["SettlementPointPrice"], errors="coerce")
    
    # Deduplicate (in case of overlapping downloads)
    result = df[["timestamp", "lmp_price_usd_mwh"]].dropna()
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    result["region"] = "ERCOT"
    
    # Save
    outpath = os.path.join(OUTPUT_DIR, "real_lmp_ERCOT_2020_2025.csv")
    result.to_csv(outpath, index=False)
    
    print(f"\n{'='*60}")
    print(f"ERCOT REAL DAM PRICES — FINAL")
    print(f"{'='*60}")
    print(f"  Total hourly prices: {len(result):,}")
    print(f"  Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"  Unique days: {result['timestamp'].dt.date.nunique()}")
    print(f"  Price range: ${result['lmp_price_usd_mwh'].min():.2f} to ${result['lmp_price_usd_mwh'].max():.2f} /MWh")
    print(f"  Mean: ${result['lmp_price_usd_mwh'].mean():.2f} /MWh")
    print(f"  Median: ${result['lmp_price_usd_mwh'].median():.2f} /MWh")
    print(f"  Saved: {outpath}")
