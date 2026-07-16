"""
Test v3 — Multiple approaches to get REAL electricity prices.
Strategy:
1. CAISO OASIS with correct parameters (PRC_INTVL_LMP, version=3)
2. EIA API (user has key) for wholesale prices  
3. ERCOT via direct file download
4. PJM API Portal (may need user to register for free key)
"""

import requests
import zipfile
import pandas as pd
from io import BytesIO, StringIO
import xml.etree.ElementTree as ET
import os

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

print("=" * 60)
print("TESTING DATA ACCESS — V3")
print("=" * 60)

# =====================================================
# 1. CAISO with correct endpoint (PRC_INTVL_LMP, version=3)
# =====================================================
print("\n[1] CAISO — PRC_INTVL_LMP with version=3...")
url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    "queryname": "PRC_INTVL_LMP",
    "market_run_id": "RTM",
    "startdatetime": "20240101T00:00-0000",
    "enddatetime": "20240102T00:00-0000",
    "version": "3",
    "node": "TH_NP15_GEN-APND",
    "resultformat": "6",
}
resp = requests.get(url, params=params, timeout=60)
print(f"  Status: {resp.status_code}, Size: {len(resp.content)} bytes")
if resp.status_code == 200 and len(resp.content) > 200:
    try:
        z = zipfile.ZipFile(BytesIO(resp.content))
        for fname in z.namelist():
            print(f"  File: {fname}")
            content = z.read(fname).decode("utf-8")
            if fname.endswith(".csv"):
                df = pd.read_csv(StringIO(content))
                print(f"  Rows: {len(df)}, Cols: {df.columns.tolist()}")
                if len(df) > 0:
                    print(df.head(2).to_string())
            elif fname.endswith(".xml"):
                # Check if it's an error
                if "ERR_CODE" in content:
                    root = ET.fromstring(content)
                    ns = {"m": "http://www.caiso.com/soa/OASISReport_v1.xsd"}
                    err = root.find(".//m:ERR_DESC", ns)
                    if err is not None:
                        print(f"  Error: {err.text}")
                    else:
                        print(f"  XML (first 500): {content[:500]}")
                else:
                    print(f"  XML data found, size: {len(content)}")
    except zipfile.BadZipFile:
        print(f"  Not a zip. Content: {resp.content[:300]}")

# =====================================================
# 2. EIA API — Wholesale Electricity Prices (hourly)
# EIA has hourly wholesale prices for major hubs!
# =====================================================
print("\n[2] EIA API — checking available wholesale price data...")

# EIA APIv2 — hourly wholesale electricity prices
# Series: EBA (Electricity Balancing Authority)
url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "PJM",
    "facets[type][]": "D",  # Demand
    "start": "2024-01-01T00",
    "end": "2024-01-02T00",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": "25",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  EIA region-data Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    if "response" in data and "data" in data["response"]:
        items = data["response"]["data"]
        print(f"  Got {len(items)} items")
        if items:
            print(f"  Sample: {items[0]}")

# Check if EIA has spot prices / wholesale prices
print("\n[2b] EIA API — checking spot/wholesale price endpoint...")
url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
params = {
    "api_key": EIA_API_KEY,
    "frequency": "daily",
    "data[0]": "value",
    "facets[respondent][]": "PJM",
    "facets[type][]": "DF",  # Day-ahead forecasted demand
    "start": "2024-01-01",
    "end": "2024-01-07",
    "length": "10",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  EIA daily Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    if "response" in data:
        resp_data = data["response"]
        if "data" in resp_data:
            print(f"  Items: {len(resp_data['data'])}")
            if resp_data["data"]:
                print(f"  Sample: {resp_data['data'][0]}")

# Try EIA wholesale electricity market data
print("\n[2c] EIA API — wholesale electricity market prices...")
url = "https://api.eia.gov/v2/electricity/rto/wholesale-prices/data/"
params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "start": "2024-01-01T00",
    "end": "2024-01-02T00",
    "length": "25",
}
resp = requests.get(url, params=params, timeout=30)
print(f"  EIA wholesale-prices Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    if "response" in data and "data" in data["response"]:
        items = data["response"]["data"]
        print(f"  Got {len(items)} items")
        if items:
            print(f"  Sample: {items[0]}")
    elif "error" in data:
        print(f"  Error: {data['error']}")
else:
    print(f"  Response: {resp.text[:300]}")

# =====================================================
# 3. Try EIA spot prices endpoint
# =====================================================
print("\n[3] EIA API — exploring available routes...")
url = "https://api.eia.gov/v2/electricity/rto/"
params = {"api_key": EIA_API_KEY}
resp = requests.get(url, params=params, timeout=30)
print(f"  Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    if "response" in data:
        routes = data["response"].get("routes", [])
        print(f"  Available routes:")
        for r in routes:
            print(f"    - {r.get('id', '?')}: {r.get('name', '?')} ({r.get('description', '')})")

# =====================================================
# 4. ERCOT file download — fix the doc ID
# =====================================================
print("\n[4] ERCOT — fetching with correct URL format...")
# The report 13060 = Historical DAM Load Zone and Hub Prices
url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
params = {
    "reportTypeId": "13060",
    "controlsSearch": "2024",
}
resp = requests.get(url, params=params, timeout=30)
if resp.status_code == 200:
    data = resp.json()
    docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    print(f"  Found {len(docs)} documents for 2024")
    if docs:
        doc = docs[0]["Document"]
        doc_id = doc["DocID"]
        construct_name = doc["ConstructedName"]
        print(f"  First: {construct_name} (ID: {doc_id})")
        
        # Try correct download URL format
        download_url = f"https://www.ercot.com/misdownload/servlets/mirDownload?miession=&dession=&showHTMLError=false&requestfilter=All&requestType=File&requestAction=download&docid={doc_id}"
        resp2 = requests.get(download_url, timeout=60, allow_redirects=True)
        print(f"  Download: status={resp2.status_code}, size={len(resp2.content)} bytes")
        print(f"  Content-Type: {resp2.headers.get('content-type', 'unknown')}")
        
        if len(resp2.content) > 500:
            try:
                z = zipfile.ZipFile(BytesIO(resp2.content))
                for fname in z.namelist():
                    print(f"    ZIP file: {fname}")
                    if fname.endswith(".csv"):
                        csv_content = z.read(fname).decode("utf-8")
                        df = pd.read_csv(StringIO(csv_content))
                        print(f"    Rows: {len(df)}, Cols: {df.columns.tolist()}")
                        if len(df) > 0:
                            print(f"    Sample:\n{df.head(2).to_string()}")
                        break
            except zipfile.BadZipFile:
                print(f"    Not a ZIP file. First 200 bytes: {resp2.content[:200]}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
