"""Test v2 — adjusted API calls based on what we learned."""

import requests
import zipfile
import pandas as pd
from io import BytesIO, StringIO
import xml.etree.ElementTree as ET

print("=" * 60)
print("TESTING ISO API ACCESS — V2")
print("=" * 60)

# --- CAISO: Parse the XML format ---
print("\n[1] CAISO OASIS — trying XML parse...")
url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    "queryname": "PRC_LMP",
    "market_run_id": "RTM",
    "startdatetime": "20240101T08:00-0000",
    "enddatetime": "20240102T08:00-0000",
    "node": "TH_NP15_GEN-APND",
    "resultformat": "6",  # 6 = CSV
    "version": "1",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  Status: {resp.status_code}, Size: {len(resp.content)} bytes")

if resp.status_code == 200:
    z = zipfile.ZipFile(BytesIO(resp.content))
    for fname in z.namelist():
        print(f"  File: {fname}")
        content = z.read(fname).decode("utf-8")
        if fname.endswith(".xml"):
            # Parse XML
            print(f"  XML content (first 1000 chars):")
            print(f"  {content[:1000]}")
        elif fname.endswith(".csv"):
            df = pd.read_csv(StringIO(content))
            print(f"  CSV rows: {len(df)}")
            print(f"  Columns: {df.columns.tolist()}")

# Try CAISO with different resultformat
print("\n[1b] CAISO OASIS — resultformat=2 (flat csv)...")
params["resultformat"] = "2"
resp = requests.get(url, params=params, timeout=30)
print(f"  Status: {resp.status_code}, Size: {len(resp.content)} bytes")
if resp.status_code == 200:
    z = zipfile.ZipFile(BytesIO(resp.content))
    for fname in z.namelist():
        print(f"  File: {fname}")
        content = z.read(fname).decode("utf-8")
        if fname.endswith(".csv"):
            df = pd.read_csv(StringIO(content))
            print(f"  Rows: {len(df)}, Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(df.head(2).to_string())


# --- PJM: Check what dataminer2 actually returns ---
print("\n[2] PJM Data Miner 2 — checking response format...")
url = "https://dataminer2.pjm.com/feed/rt_hrl_lmps"
params = {
    "startRow": "1",
    "rowCount": "24",
    "datetime_beginning_ept": "1/1/2024 00:00to1/2/2024 00:00",
    "pnode_id": "51217",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  Status: {resp.status_code}")
print(f"  Content-Type: {resp.headers.get('content-type', 'unknown')}")
print(f"  Size: {len(resp.content)} bytes")
print(f"  First 500 chars: {resp.text[:500]}")


# --- ERCOT: Download actual price file ---
print("\n[3] ERCOT — downloading actual price ZIP...")
# First get the document list
url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
params = {
    "reportTypeId": "13060",  # Historical DAM Load Zone and Hub Prices
    "controlsSearch": "202401",
}
resp = requests.get(url, params=params, timeout=30)
data = resp.json()
docs = data["ListDocsByRptTypeRes"]["DocumentList"]
print(f"  Found {len(docs)} documents")

# Get the first document (most recent)
if docs:
    doc = docs[0]["Document"]
    doc_id = doc["DocID"]
    construct_name = doc["ConstructedName"]
    print(f"  Downloading: {construct_name}")
    
    # Download the actual file
    download_url = f"https://www.ercot.com/misdownload/servlets/mirDownload?dession=&miession=&def=true&docid={doc_id}"
    resp = requests.get(download_url, timeout=60)
    print(f"  Download status: {resp.status_code}, Size: {len(resp.content)} bytes")
    
    if resp.status_code == 200 and len(resp.content) > 100:
        try:
            z = zipfile.ZipFile(BytesIO(resp.content))
            for fname in z.namelist():
                print(f"  ZIP contains: {fname}")
                if fname.endswith(".csv") or fname.endswith(".xlsx"):
                    content = z.read(fname)
                    if fname.endswith(".csv"):
                        df = pd.read_csv(BytesIO(content))
                    else:
                        df = pd.read_excel(BytesIO(content))
                    print(f"  Rows: {len(df)}, Columns: {df.columns.tolist()}")
                    if len(df) > 0:
                        print(f"  Sample:\n{df.head(2).to_string()}")
        except zipfile.BadZipFile:
            print(f"  Not a ZIP. Content-Type: {resp.headers.get('content-type')}")
            print(f"  First 200 bytes: {resp.content[:200]}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
