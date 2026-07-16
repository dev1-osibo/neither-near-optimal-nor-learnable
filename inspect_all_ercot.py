"""
INSPECT EVERY ERCOT FILE — no assumptions, full audit.
Reports: entry count, date range, columns, hubs present, any anomalies.
"""

import zipfile
import pandas as pd
import os
from io import BytesIO

DATASETS_DIR = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\datasets"

ercot_zips = sorted([
    f for f in os.listdir(DATASETS_DIR) 
    if f.startswith("cdr.np4-190") and f.endswith(".zip")
])

print(f"{'='*70}")
print(f"FULL AUDIT OF ALL {len(ercot_zips)} ERCOT FILES")
print(f"{'='*70}\n")

grand_total_days = 0
all_dates_found = []

for i, zf in enumerate(ercot_zips):
    path = os.path.join(DATASETS_DIR, zf)
    size_mb = os.path.getsize(path) / (1024*1024)
    
    print(f"{'─'*70}")
    print(f"FILE {i+1}/{len(ercot_zips)}: {zf}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"{'─'*70}")
    
    try:
        outer = zipfile.ZipFile(path)
        entries = outer.namelist()
        print(f"  Entries: {len(entries)}")
        
        # Check first and last entry to get date range
        dates_in_file = []
        columns_found = None
        hubs_found = None
        row_counts = []
        
        # Check first entry
        first_entry = entries[0]
        last_entry = entries[-1]
        
        for check_entry in [first_entry, last_entry]:
            content = outer.read(check_entry)
            if check_entry.endswith(".zip"):
                try:
                    inner = zipfile.ZipFile(BytesIO(content))
                    for csv_name in inner.namelist():
                        if csv_name.endswith(".csv"):
                            df = pd.read_csv(BytesIO(inner.read(csv_name)))
                            row_counts.append(len(df))
                            if columns_found is None:
                                columns_found = df.columns.tolist()
                            if "DeliveryDate" in df.columns:
                                dates_in_file.append(df["DeliveryDate"].iloc[0])
                            if "SettlementPoint" in df.columns and hubs_found is None:
                                sp = df["SettlementPoint"].unique()
                                hubs_found = [s for s in sp if "HB_" in str(s)]
                except Exception as e:
                    print(f"  ERROR reading inner zip {check_entry}: {e}")
            elif check_entry.endswith(".csv"):
                try:
                    df = pd.read_csv(BytesIO(content))
                    row_counts.append(len(df))
                    if columns_found is None:
                        columns_found = df.columns.tolist()
                    if "DeliveryDate" in df.columns:
                        dates_in_file.append(df["DeliveryDate"].iloc[0])
                    if "SettlementPoint" in df.columns and hubs_found is None:
                        sp = df["SettlementPoint"].unique()
                        hubs_found = [s for s in sp if "HB_" in str(s)]
                except Exception as e:
                    print(f"  ERROR reading CSV {check_entry}: {e}")
        
        # Also spot-check a middle entry
        mid_idx = len(entries) // 2
        mid_entry = entries[mid_idx]
        mid_content = outer.read(mid_entry)
        if mid_entry.endswith(".zip"):
            try:
                inner = zipfile.ZipFile(BytesIO(mid_content))
                for csv_name in inner.namelist():
                    if csv_name.endswith(".csv"):
                        df = pd.read_csv(BytesIO(inner.read(csv_name)))
                        row_counts.append(len(df))
                        if "DeliveryDate" in df.columns:
                            dates_in_file.append(df["DeliveryDate"].iloc[0])
            except:
                pass
        
        # Report
        print(f"  Columns: {columns_found}")
        print(f"  Hubs present: {hubs_found}")
        print(f"  Rows per daily file: {row_counts}")
        print(f"  Dates found (first/mid/last): {dates_in_file}")
        
        if dates_in_file:
            all_dates_found.extend(dates_in_file)
        
        grand_total_days += len(entries)
        print(f"  ✓ OK — {len(entries)} daily files")
        
    except Exception as e:
        print(f"  CRITICAL ERROR: {e}")

# Final summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total files: {len(ercot_zips)}")
print(f"Total daily entries: {grand_total_days}")
print(f"Expected hourly HB_HOUSTON rows: {grand_total_days * 24:,}")
print(f"\nAll dates sampled (chronological):")
# Parse and sort dates
parsed_dates = []
for d in all_dates_found:
    try:
        parsed_dates.append(pd.to_datetime(d))
    except:
        parsed_dates.append(None)
parsed_dates = sorted([d for d in parsed_dates if d is not None])
if parsed_dates:
    print(f"  Earliest: {parsed_dates[0]}")
    print(f"  Latest: {parsed_dates[-1]}")
    print(f"  All sampled: {[d.strftime('%Y-%m-%d') for d in parsed_dates]}")
