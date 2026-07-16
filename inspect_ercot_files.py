"""
INSPECT ERCOT files — no assumptions, just report what's inside.
"""

import zipfile
import pandas as pd
import os
from io import BytesIO

DATASETS_DIR = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\datasets"

# Find all ERCOT NP4-190 ZIP files
ercot_zips = sorted([
    f for f in os.listdir(DATASETS_DIR) 
    if f.startswith("cdr.np4-190") and f.endswith(".zip")
])

print(f"Found {len(ercot_zips)} ERCOT ZIP files\n")

for i, zf in enumerate(ercot_zips):
    path = os.path.join(DATASETS_DIR, zf)
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"{'='*70}")
    print(f"FILE {i+1}: {zf} ({size_mb:.1f} MB)")
    print(f"{'='*70}")
    
    try:
        outer = zipfile.ZipFile(path)
        names = outer.namelist()
        print(f"  Total entries: {len(names)}")
        print(f"  First 3 entries:")
        for n in names[:3]:
            info = outer.getinfo(n)
            print(f"    {n} ({info.file_size:,} bytes)")
        print(f"  Last 3 entries:")
        for n in names[-3:]:
            info = outer.getinfo(n)
            print(f"    {n} ({info.file_size:,} bytes)")
        
        # Open the FIRST inner entry to see its structure
        first_name = names[0]
        first_content = outer.read(first_name)
        
        print(f"\n  --- Inspecting first entry: {first_name} ---")
        
        if first_name.endswith(".zip"):
            # It's a nested zip
            inner = zipfile.ZipFile(BytesIO(first_content))
            inner_files = inner.namelist()
            print(f"  Inner zip contains: {inner_files}")
            for inner_f in inner_files:
                if inner_f.endswith(".csv"):
                    df = pd.read_csv(BytesIO(inner.read(inner_f)))
                    print(f"  CSV: {inner_f}")
                    print(f"    Rows: {len(df)}")
                    print(f"    Columns: {df.columns.tolist()}")
                    print(f"    Dtypes:")
                    for col in df.columns:
                        print(f"      {col}: {df[col].dtype} | sample: {df[col].iloc[0]}")
                    # Check what settlement points exist
                    if "SettlementPoint" in df.columns:
                        sp = df["SettlementPoint"].unique()
                        hubs = [s for s in sp if "HB_" in str(s)]
                        print(f"    Unique SettlementPoints: {len(sp)}")
                        print(f"    Hubs: {hubs}")
                    print(f"    First 2 rows:")
                    print(df.head(2).to_string(index=False))
                    print(f"    Last 2 rows:")
                    print(df.tail(2).to_string(index=False))
        elif first_name.endswith(".csv"):
            # Direct CSV
            df = pd.read_csv(BytesIO(first_content))
            print(f"  Direct CSV: {first_name}")
            print(f"    Rows: {len(df)}")
            print(f"    Columns: {df.columns.tolist()}")
            for col in df.columns:
                print(f"      {col}: {df[col].dtype} | sample: {df[col].iloc[0]}")
        else:
            print(f"  Unknown format: {first_name}")
            print(f"  First 200 bytes: {first_content[:200]}")
    
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print()
    
    # Only inspect first 2 files in detail to save time
    if i >= 1:
        print(f"... (skipping detailed inspection of remaining {len(ercot_zips)-2} files)")
        print(f"    Remaining files:")
        for zf2 in ercot_zips[2:]:
            size = os.path.getsize(os.path.join(DATASETS_DIR, zf2)) / (1024*1024)
            # Quick check: count entries
            try:
                z = zipfile.ZipFile(os.path.join(DATASETS_DIR, zf2))
                print(f"    {zf2}: {size:.1f} MB, {len(z.namelist())} entries")
            except:
                print(f"    {zf2}: {size:.1f} MB, could not open")
        break
