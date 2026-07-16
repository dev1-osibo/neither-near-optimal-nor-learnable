"""Test CAISO PRC_LMP (hourly) for older years."""

import requests
import zipfile
import pandas as pd
import time
from io import BytesIO, StringIO

url = "http://oasis.caiso.com/oasisapi/SingleZip"

# Try PRC_LMP (hourly resolution — different from PRC_INTVL_LMP which is 5-min)
test_configs = [
    # (queryname, market_run_id, version, start, label)
    ("PRC_LMP", "DAM", "12", "2020-01-01", "PRC_LMP DAM v12 2020"),
    ("PRC_LMP", "DAM", "3", "2020-01-01", "PRC_LMP DAM v3 2020"),
    ("PRC_LMP", "DAM", "1", "2020-01-01", "PRC_LMP DAM v1 2020"),
    ("PRC_LMP", "RTM", "3", "2020-01-01", "PRC_LMP RTM v3 2020"),
    ("PRC_HASP_LMP", "HASP", "3", "2020-01-01", "HASP LMP 2020"),
    ("PRC_LMP", "DAM", "12", "2022-01-01", "PRC_LMP DAM v12 2022"),
    ("PRC_LMP", "DAM", "12", "2023-01-01", "PRC_LMP DAM v12 2023"),
]

for queryname, market, version, start_str, label in test_configs:
    d = pd.Timestamp(start_str)
    e = d + pd.Timedelta(days=7)
    
    params = {
        "queryname": queryname,
        "market_run_id": market,
        "startdatetime": d.strftime("%Y%m%dT08:00-0000"),
        "enddatetime": e.strftime("%Y%m%dT08:00-0000"),
        "version": version,
        "node": "TH_NP15_GEN-APND",
        "resultformat": "6",
    }
    resp = requests.get(url, params=params, timeout=60)
    
    result = "?"
    if resp.status_code == 200 and len(resp.content) > 500:
        try:
            z = zipfile.ZipFile(BytesIO(resp.content))
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    df = pd.read_csv(StringIO(z.read(fname).decode("utf-8")))
                    if "LMP_TYPE" in df.columns:
                        lmp = df[df["LMP_TYPE"] == "LMP"]
                    else:
                        lmp = df
                    result = f"CSV: {len(lmp)} rows"
                    if not lmp.empty and "VALUE" in lmp.columns:
                        vals = pd.to_numeric(lmp["VALUE"], errors="coerce")
                        result += f", avg ${vals.mean():.1f}/MWh"
                    if not lmp.empty:
                        print(f"  Columns: {df.columns.tolist()[:8]}")
                elif fname.endswith(".xml"):
                    content = z.read(fname).decode("utf-8")
                    if "ERR_CODE" in content:
                        result = "NO DATA"
                    elif "INVALID" in fname.upper():
                        result = "INVALID REQUEST"
                    else:
                        result = f"XML ({len(content)} chars)"
        except zipfile.BadZipFile:
            result = "Bad ZIP"
    elif resp.status_code == 429:
        result = "RATE LIMITED"
        time.sleep(30)
    else:
        result = f"HTTP {resp.status_code} ({len(resp.content)} bytes)"
    
    print(f"  [{label}]: {result}")
    time.sleep(7)
