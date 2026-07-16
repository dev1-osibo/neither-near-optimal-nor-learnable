"""Test which years CAISO has data for."""

import requests
import zipfile
import pandas as pd
import time
from io import BytesIO, StringIO

url = "http://oasis.caiso.com/oasisapi/SingleZip"

test_starts = [
    "2020-06-01", "2021-01-01", "2022-01-01", 
    "2023-01-01", "2023-06-01", "2023-12-01",
    "2024-01-01", "2024-06-01", "2025-01-01",
]

for start_str in test_starts:
    d = pd.Timestamp(start_str)
    e = d + pd.Timedelta(days=7)
    
    params = {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "startdatetime": d.strftime("%Y%m%dT00:00-0000"),
        "enddatetime": e.strftime("%Y%m%dT00:00-0000"),
        "version": "3",
        "node": "TH_NP15_GEN-APND",
        "resultformat": "6",
    }
    resp = requests.get(url, params=params, timeout=60)
    
    result = "?"
    if resp.status_code == 200 and len(resp.content) > 1000:
        try:
            z = zipfile.ZipFile(BytesIO(resp.content))
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    df = pd.read_csv(StringIO(z.read(fname).decode("utf-8")))
                    lmp = df[df["LMP_TYPE"] == "LMP"]
                    vals = pd.to_numeric(lmp["VALUE"], errors="coerce")
                    result = f"OK - {len(lmp)} rows, avg ${vals.mean():.1f}/MWh"
                elif fname.endswith(".xml"):
                    content = z.read(fname).decode("utf-8")
                    if "ERR_CODE" in content:
                        result = "NO DATA (API error)"
                    else:
                        result = "XML (unknown)"
        except zipfile.BadZipFile:
            result = "Bad ZIP"
    elif resp.status_code == 429:
        result = "RATE LIMITED"
    else:
        result = f"HTTP {resp.status_code} ({len(resp.content)} bytes)"
    
    print(f"  {start_str}: {result}")
    time.sleep(7)
