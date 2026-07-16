"""Check what's actually in the ERCOT 2020 file."""

import zipfile
import pandas as pd
from io import BytesIO

path = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\rpt.00013060.0000000000000000.20210101.080809625.DAMLZHBSPP_2020.zip"
z = zipfile.ZipFile(path)

for fname in z.namelist():
    print(f"File: {fname}")
    df = pd.read_excel(BytesIO(z.read(fname)), engine="openpyxl")
    print(f"Total rows: {len(df)}")
    
    sp = df["Settlement Point"].unique()
    print(f"Settlement Points ({len(sp)}): {sp.tolist()}")
    print(f"Rows per point: {len(df) // len(sp)}")
    
    # Check HB_HOUSTON date range
    hb = df[df["Settlement Point"] == "HB_HOUSTON"].copy()
    print(f"\nHB_HOUSTON: {len(hb)} rows")
    hb["date"] = pd.to_datetime(hb["Delivery Date"])
    print(f"Date range: {hb['date'].min()} to {hb['date'].max()}")
    print(f"Unique dates: {hb['date'].nunique()}")
    
    # Show monthly coverage
    hb["month"] = hb["date"].dt.month
    print(f"\nRows per month:")
    print(hb.groupby("month").size())
