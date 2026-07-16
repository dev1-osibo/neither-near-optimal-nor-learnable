"""Test CAISO API with different date ranges to find working pattern."""

import requests
import zipfile
import pandas as pd
import time
from io import BytesIO, StringIO

url = "http://oasis.caiso.com/oasisapi/SingleZip"

def test_caiso(start, end, label):
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "startdatetime": start,
        "enddatetime": end,
        "version": "3",
        "node": "TH_NP15_GEN-APND",
        "resultformat": "6",
    }
    resp = requests.get(url, params=params, timeout=60)
    print(f"[{label}] status={resp.status_code}, size={len(resp.content)} bytes")
    
    if resp.status_code == 200 and len(resp.content) > 200:
        try:
            z = zipfile.ZipFile(BytesIO(resp.content))
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    df = pd.read_csv(StringIO(z.read(fname).decode("utf-8")))
                    df_lmp = df[df["LMP_TYPE"] == "LMP"]
                    print(f"  ✓ CSV: {len(df_lmp)} LMP rows")
                    if not df_lmp.empty:
                        vals = pd.to_numeric(df_lmp["VALUE"], errors="coerce")
                        print(f"  Price: ${vals.mean():.2f}/MWh avg, ${vals.min():.2f}-${vals.max():.2f}")
                    return True
                elif fname.endswith(".xml"):
                    content = z.read(fname).decode("utf-8")
                    if "ERR_CODE" in content:
                        print(f"  ✗ Error: No data for period")
                    else:
                        print(f"  ? XML but no error tag")
                    return False
        except zipfile.BadZipFile:
            print(f"  ✗ Bad ZIP")
            return False
    elif resp.status_code == 429:
        print(f"  ✗ Rate limited")
        return False
    return False

# Test 1: 1 day (we know this works)
print("=" * 50)
test_caiso("20240101T00:00-0000", "20240102T00:00-0000", "1 day")
time.sleep(6)

# Test 2: 7 days
print()
test_caiso("20240101T00:00-0000", "20240108T00:00-0000", "7 days")
time.sleep(6)

# Test 3: 31 days (the limit per docs)
print()
test_caiso("20240101T00:00-0000", "20240201T00:00-0000", "31 days")
time.sleep(6)

# Test 4: Try DAM instead of RTM
print()
params_dam = {
    "queryname": "PRC_LMP",
    "market_run_id": "DAM",
    "startdatetime": "20240101T00:00-0000",
    "enddatetime": "20240108T00:00-0000",
    "version": "3",
    "node": "TH_NP15_GEN-APND",
    "resultformat": "6",
}
resp = requests.get(url, params=params_dam, timeout=60)
print(f"[DAM 7 days] status={resp.status_code}, size={len(resp.content)} bytes")
if resp.status_code == 200 and len(resp.content) > 200:
    try:
        z = zipfile.ZipFile(BytesIO(resp.content))
        for fname in z.namelist():
            print(f"  File: {fname}")
            if fname.endswith(".csv"):
                df = pd.read_csv(StringIO(z.read(fname).decode("utf-8")))
                df_lmp = df[df["LMP_TYPE"] == "LMP"]
                print(f"  ✓ DAM CSV: {len(df_lmp)} LMP rows")
                if not df_lmp.empty:
                    vals = pd.to_numeric(df_lmp["VALUE"], errors="coerce")
                    print(f"  Price: ${vals.mean():.2f}/MWh avg")
            elif fname.endswith(".xml"):
                content = z.read(fname).decode("utf-8")
                if "ERR_CODE" in content:
                    print(f"  ✗ Error")
                else:
                    print(f"  XML data (not error)")
    except:
        pass
