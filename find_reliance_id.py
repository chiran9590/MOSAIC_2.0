import pandas as pd
import requests
from pathlib import Path

url = "https://images.dhan.co/api-data/api-scrip-master.csv"

print("Downloading Dhan instrument list...")

response = requests.get(url, timeout=60)
response.raise_for_status()

Path("data/instruments").mkdir(parents=True, exist_ok=True)

csv_path = Path("data/instruments/dhan_instruments.csv")
csv_path.write_bytes(response.content)

df = pd.read_csv(csv_path)

matches = df[
    (df["SEM_EXM_EXCH_ID"] == "NSE")
    & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")
    & (df["SEM_TRADING_SYMBOL"] == "RELIANCE")
]

if matches.empty:
    print("RELIANCE was not found. Check the instrument list.")
else:
    print("\nRELIANCE instrument details:")
    print(matches.to_string(index=False))

    security_id = str(matches.iloc[0]["SEM_SMST_SECURITY_ID"])
    print("\nRELIANCE Security ID:", security_id)