"""Test ERCOT Public API."""

import requests

# Try ERCOT Public API (data from Dec 2023+)
url = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
params = {
    "deliveryDateFrom": "2024-01-01",
    "deliveryDateTo": "2024-01-02",
    "settlementPoint": "HB_HOUSTON",
}
headers = {"Accept": "application/json"}
resp = requests.get(url, params=params, headers=headers, timeout=30)
print(f"ERCOT Public API: status={resp.status_code}, size={len(resp.content)}")
if resp.status_code == 200:
    data = resp.json()
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        if "data" in data:
            items = data["data"]
            print(f"Items: {len(items)}")
            if items:
                print(f"Sample: {items[0]}")
    elif isinstance(data, list):
        print(f"Items: {len(data)}")
        if data:
            print(f"Sample: {data[0]}")
else:
    print(f"Response: {resp.text[:500]}")

# Try data.ercot.com
print()
url2 = "https://data.ercot.com/data-product-archive/NP4-190-CD"
resp2 = requests.get(url2, timeout=30)
print(f"ERCOT data archive: status={resp2.status_code}")
ct = resp2.headers.get("content-type", "unknown")
print(f"Content-type: {ct}")
if resp2.status_code == 200:
    print(f"First 500: {resp2.text[:500]}")
