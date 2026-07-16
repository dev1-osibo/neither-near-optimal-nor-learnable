"""Check the new 9th ERCOT file (likely 2020 data)."""

import zipfile
import pandas as pd
from io import BytesIO

path = r"c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\datasets\cdr.np4-190-cd.00012331.20260622.195807.365.zip"
z = zipfile.ZipFile(path)
entries = z.namelist()
print(f"Entries: {len(entries)}")
print(f"First entry: {entries[0]}")
print(f"Last entry: {entries[-1]}")

# Check first entry
print("\n--- First entry ---")
inner = zipfile.ZipFile(BytesIO(z.read(entries[0])))
for csv_name in inner.namelist():
    if csv_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(inner.read(csv_name)))
        print(f"Columns: {df.columns.tolist()}")
        print(f"Rows: {len(df)}")
        date_val = df["DeliveryDate"].iloc[0]
        print(f"First date: {date_val}")
        sp = df["SettlementPoint"].unique()
        hubs = [s for s in sp if "HB_" in str(s)]
        print(f"Hubs: {hubs}")
        print(f"Total settlement points: {len(sp)}")

# Check last entry
print("\n--- Last entry ---")
inner2 = zipfile.ZipFile(BytesIO(z.read(entries[-1])))
for csv_name in inner2.namelist():
    if csv_name.endswith(".csv"):
        df2 = pd.read_csv(BytesIO(inner2.read(csv_name)))
        date_val2 = df2["DeliveryDate"].iloc[0]
        print(f"Last date: {date_val2}")
        print(f"Rows: {len(df2)}")

# Check middle entry
print("\n--- Middle entry ---")
mid = entries[len(entries)//2]
inner3 = zipfile.ZipFile(BytesIO(z.read(mid)))
for csv_name in inner3.namelist():
    if csv_name.endswith(".csv"):
        df3 = pd.read_csv(BytesIO(inner3.read(csv_name)))
        date_val3 = df3["DeliveryDate"].iloc[0]
        print(f"Mid date: {date_val3}")

print(f"\n--- Summary ---")
print(f"This file covers: {date_val} to {date_val2}")
print(f"Entries (days): {len(entries)}")
