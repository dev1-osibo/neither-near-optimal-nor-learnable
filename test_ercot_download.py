"""Test ERCOT download with different URL patterns."""

import requests
import zipfile
import pandas as pd
from io import BytesIO, StringIO

print("Testing ERCOT downloads...")

# Get document list for 2024
url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
params = {"reportTypeId": "13060", "controlsSearch": "2024"}
resp = requests.get(url, params=params, timeout=30)
data = resp.json()
docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
print(f"Found {len(docs)} documents")

for doc_entry in docs[:5]:
    doc = doc_entry.get("Document", doc_entry)
    print(f"  - {doc.get('FriendlyName', '?')} | ID: {doc.get('DocID')} | Size: {doc.get('ContentSize', '?')} bytes")

# Try different download URL patterns
if docs:
    # Find 2024 specifically
    for doc_entry in docs:
        doc = doc_entry.get("Document", doc_entry)
        if "2024" in doc.get("FriendlyName", ""):
            doc_id = doc["DocID"]
            print(f"\nTrying to download: {doc.get('FriendlyName')} (ID: {doc_id})")
            
            # Pattern 1
            url1 = f"https://www.ercot.com/misdownload/servlets/mirDownload?dession=&miession=&def=true&docid={doc_id}"
            resp1 = requests.get(url1, timeout=60, allow_redirects=True)
            print(f"  Pattern 1: status={resp1.status_code}, size={len(resp1.content)}")
            
            # Pattern 2 — with requestType
            url2 = f"https://www.ercot.com/misdownload/servlets/mirDownload?miession=&dession=&showHTMLError=false&requestfilter=All&requestType=File&requestAction=download&docid={doc_id}"
            resp2 = requests.get(url2, timeout=60, allow_redirects=True)
            print(f"  Pattern 2: status={resp2.status_code}, size={len(resp2.content)}")
            
            # Pattern 3 — construct name
            construct = doc.get("ConstructedName", "")
            url3 = f"https://www.ercot.com/misapp/GetReports.do?reportTypeId=13060&reportTitle=Historical%20DAM%20Load%20Zone%20and%20Hub%20Prices&showHTMLError=false&mimession="
            resp3 = requests.get(url3, timeout=60, allow_redirects=True)
            print(f"  Pattern 3 (reports page): status={resp3.status_code}, size={len(resp3.content)}")
            
            # Check if any response is a valid zip
            for i, resp in enumerate([resp1, resp2, resp3], 1):
                if len(resp.content) > 1000:
                    try:
                        z = zipfile.ZipFile(BytesIO(resp.content))
                        print(f"  Pattern {i} IS a valid ZIP! Contents: {z.namelist()[:3]}")
                        # Read first CSV
                        for fname in z.namelist():
                            if fname.endswith(".csv"):
                                df = pd.read_csv(BytesIO(z.read(fname)))
                                print(f"    {fname}: {len(df)} rows, cols={df.columns.tolist()[:5]}")
                                if len(df) > 0:
                                    print(f"    First row: {df.iloc[0].to_dict()}")
                                break
                    except zipfile.BadZipFile:
                        # Maybe direct CSV?
                        try:
                            df = pd.read_csv(StringIO(resp.text[:10000]))
                            print(f"  Pattern {i} is direct CSV: {len(df)} rows")
                        except:
                            pass
            break
