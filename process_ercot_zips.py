"""Process downloaded ERCOT DAM Hub/Load Zone price ZIP files."""

import zipfile
import pandas as pd
import os
from io import BytesIO

data_dir = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration"
output_dir = os.path.join(data_dir, "data")

# Find all ERCOT ZIP files
zips = [f for f in os.listdir(data_dir) if f.endswith(".zip") and "DAMLZHBSPP" in f]
zips.sort()
print(f"Found {len(zips)} ERCOT ZIP files\n")

all_data = []
for zf in zips:
    path = os.path.join(data_dir, zf)
    z = zipfile.ZipFile(path)
    for fname in z.namelist():
        if fname.endswith(".csv") or fname.endswith(".xlsx"):
            content = BytesIO(z.read(fname))
            if fname.endswith(".xlsx"):
                df = pd.read_excel(content, engine="openpyxl")
            else:
                df = pd.read_csv(content)
            all_data.append(df)
            print(f"  {zf}")
            print(f"    -> {fname}: {len(df)} rows")
            if len(all_data) == 1:
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    First row:")
                for k, v in df.iloc[0].to_dict().items():
                    print(f"      {k}: {v}")
            break

df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal combined: {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# Show available settlement points
sp_col = [c for c in df.columns if "settlement" in c.lower() and "name" in c.lower() or "point" in c.lower()]
print(f"\nSettlement point columns: {sp_col}")

# Use actual column name from data: "Settlement Point"
sp_name_col = "Settlement Point"
sp_price_col = "Settlement Point Price"
date_col = "Delivery Date"
hour_col = "Hour Ending"

points = df[sp_name_col].unique()
hubs = [p for p in points if "HB_" in str(p)]
zones = [p for p in points if "LZ_" in str(p)]
print(f"Hubs ({len(hubs)}): {hubs}")
print(f"Load Zones ({len(zones)}): {zones[:10]}...")

# Filter to Houston Hub
hb = df[df[sp_name_col] == "HB_HOUSTON"].copy()
print(f"\nHB_HOUSTON: {len(hb):,} rows")

if not hb.empty:
    # Build proper timestamp
    hb[date_col] = pd.to_datetime(hb[date_col])
    # HourEnding is like "01:00", "02:00" ... "24:00"
    hb["hour"] = hb[hour_col].astype(str).str.replace(":00", "").str.strip().astype(int) - 1
    hb["timestamp"] = hb[date_col] + pd.to_timedelta(hb["hour"], unit="h")
    hb["lmp_price_usd_mwh"] = pd.to_numeric(hb[sp_price_col], errors="coerce")
    
    result = hb[["timestamp", "lmp_price_usd_mwh"]].dropna().sort_values("timestamp").reset_index(drop=True)
    result["region"] = "ERCOT"
    
    print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"Price range: ${result['lmp_price_usd_mwh'].min():.2f} to ${result['lmp_price_usd_mwh'].max():.2f} /MWh")
    print(f"Mean: ${result['lmp_price_usd_mwh'].mean():.2f} /MWh")
    print(f"Median: ${result['lmp_price_usd_mwh'].median():.2f} /MWh")
    
    # Save
    outpath = os.path.join(output_dir, "real_lmp_ERCOT_2020_2025.csv")
    result.to_csv(outpath, index=False)
    print(f"\n✓ Saved: {outpath} ({len(result):,} hourly prices)")
else:
    print("HB_HOUSTON not found! Available:")
    print(df[sp_name_col].value_counts().head(20))
