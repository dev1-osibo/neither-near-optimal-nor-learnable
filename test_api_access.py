"""Quick test of all three ISO API endpoints to verify access."""

import requests
import zipfile
import pandas as pd
from io import BytesIO, StringIO

print("=" * 60)
print("TESTING ISO API ACCESS")
print("=" * 60)

# --- CAISO ---
print("\n[1] CAISO OASIS API...")
url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    "queryname": "PRC_LMP",
    "market_run_id": "RTM",
    "startdatetime": "20240101T07:00-0000",
    "enddatetime": "20240102T07:00-0000",
    "node": "TH_NP15_GEN-APND",
    "resultformat": "6",
    "version": "1",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  Status: {resp.status_code}, Size: {len(resp.content)} bytes")

if resp.status_code == 200 and len(resp.content) > 100:
    try:
        z = zipfile.ZipFile(BytesIO(resp.content))
        print(f"  Files in zip: {z.namelist()}")
        for fname in z.namelist():
            if fname.endswith(".csv"):
                csv_data = z.read(fname).decode("utf-8")
                df = pd.read_csv(StringIO(csv_data))
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Rows: {len(df)}")
                if len(df) > 0:
                    print(f"  Sample:\n{df.head(2).to_string()}")
    except zipfile.BadZipFile:
        print(f"  Not a valid zip. Content preview: {resp.content[:500]}")
else:
    print(f"  Response: {resp.text[:500]}")

# --- PJM Data Miner 2 ---
print("\n[2] PJM Data Miner 2 API...")
# Try the public feed without API key
url = "https://api.pjm.com/api/v1/rt_hrl_lmps"
params = {
    "startRow": "1",
    "rowCount": "24",
    "datetime_beginning_ept": "2024-01-01T00:00:00",
    "datetime_ending_ept": "2024-01-02T00:00:00",
    "pnode_id": "51217",
}
try:
    resp = requests.get(url, params=params, timeout=30)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        if isinstance(data, dict) and "items" in data:
            print(f"  Items: {len(data['items'])}")
            if data["items"]:
                print(f"  Sample: {data['items'][0]}")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# Try PJM without subscription key header
print("\n[2b] PJM Data Miner 2 (no header)...")
url = "https://dataminer2.pjm.com/feed/rt_hrl_lmps"
params = {
    "startRow": "1",
    "rowCount": "24",
    "fields": "datetime_beginning_ept,pnode_id,pnode_name,total_lmp_rt",
    "datetime_beginning_ept": "1/1/2024 00:00to1/2/2024 00:00",
    "pnode_id": "51217",
}
try:
    resp = requests.get(url, params=params, timeout=30)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"  Items: {len(data)}")
            if data:
                print(f"  Sample: {data[0]}")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# --- ERCOT ---
print("\n[3] ERCOT Public Data...")
# ERCOT Mis portal for historical SPP
url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
params = {
    "reportTypeId": "13060",  # DAM Settlement Point Prices
    "controlsSearch": "202401",
}
try:
    resp = requests.get(url, params=params, timeout=30)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            if "ListDocsByRptTypeRes" in data:
                docs = data["ListDocsByRptTypeRes"]
                if isinstance(docs, dict) and "DocumentList" in docs:
                    doc_list = docs["DocumentList"]
                    if isinstance(doc_list, list):
                        print(f"  Documents available: {len(doc_list)}")
                        if doc_list:
                            print(f"  First doc: {doc_list[0]}")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("DONE — checking which APIs are accessible")
print("=" * 60)
