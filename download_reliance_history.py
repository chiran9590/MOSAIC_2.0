
import os
import time
import requests
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

SECURITY_ID = "2885"
EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT = "EQUITY"
INTERVAL = "5"

BASE_URL = "https://api.dhan.co/v2/charts/intraday"

IST = ZoneInfo("Asia/Kolkata")

OUTPUT_FILE = Path("data/raw/reliance_5min.csv")

# Download approximately five years of history.
HISTORY_DAYS = 5 * 365

# Keep each request below Dhan's 90-day limit.
CHUNK_DAYS = 89


# --------------------------------------------------
# 2. AUTHENTICATION
# --------------------------------------------------

def get_headers():

    if not CLIENT_ID or not ACCESS_TOKEN:
        raise ValueError(
            "Dhan credentials are missing. "
            "Please configure DHAN_CLIENT_ID and "
            "DHAN_ACCESS_TOKEN in your .env file."
        )

    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }


# --------------------------------------------------
# 3. DOWNLOAD ONE DATE RANGE
# --------------------------------------------------

def download_chunk(start_date, end_date):

    payload = {
        "securityId": SECURITY_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": INSTRUMENT,
        "interval": INTERVAL,
        "oi": False,
        "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S"),
    }

    response = requests.post(
        BASE_URL,
        headers=get_headers(),
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Dhan API error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    result = response.json()

    required_fields = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in result
    ]

    if missing_fields:
        raise RuntimeError(
            f"Unexpected Dhan response. Missing fields: "
            f"{missing_fields}. Response: {result}"
        )

    lengths = {
        len(result[field]) for field in required_fields
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Dhan returned candle arrays with different lengths."
        )

    if not result["timestamp"]:
        return pd.DataFrame()

    # Dhan timestamps are Unix timestamps.
    # Convert UTC timestamps into Indian Standard Time.
    timestamps = (
        pd.to_datetime(
            result["timestamp"],
            unit="s",
            utc=True,
        )
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": result["open"],
        "high": result["high"],
        "low": result["low"],
        "close": result["close"],
        "volume": result["volume"],
    })

    return df


# --------------------------------------------------
# 4. DOWNLOAD FULL HISTORY IN BATCHES
# --------------------------------------------------

def download_history():

    now = datetime.now(IST)

    end_date = now.replace(tzinfo=None)

    start_date = (
        now - timedelta(days=HISTORY_DAYS)
    ).replace(tzinfo=None)

    all_chunks = []

    current_start = start_date

    total_requests = 0

    print("\nMOSAIC 2.0 — RELIANCE DATA DOWNLOAD")
    print("-----------------------------------")
    print("Stock: RELIANCE")
    print("Security ID:", SECURITY_ID)
    print("Interval: 5 minutes")
    print("Start date:", start_date)
    print("End date:", end_date)

    while current_start < end_date:

        current_end = min(
            current_start + timedelta(days=CHUNK_DAYS),
            end_date,
        )

        total_requests += 1

        print(
            f"\nRequest {total_requests}: "
            f"{current_start} to {current_end}"
        )

        chunk = download_chunk(
            current_start,
            current_end,
        )

        print("Candles received:", len(chunk))

        if not chunk.empty:
            all_chunks.append(chunk)

        # Avoid sending requests too rapidly.
        time.sleep(0.5)

        current_start = current_end

    if not all_chunks:
        raise RuntimeError(
            "No historical candles were returned by Dhan."
        )

    # Combine all downloaded batches.
    df = pd.concat(
        all_chunks,
        ignore_index=True,
    )

    # Clean and sort.
    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Ensure numeric OHLCV columns.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["timestamp"] + numeric_columns
    )

    # Save the dataset.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n-----------------------------------")
    print("DOWNLOAD COMPLETED")
    print("-----------------------------------")

    print("Total API requests:", total_requests)
    print("Total candles:", len(df))
    print("First candle:", df["timestamp"].min())
    print("Last candle:", df["timestamp"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLatest five candles:")
    print(df.tail())

    print("\nDataset saved to:", OUTPUT_FILE)


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    try:
        download_history()

    except requests.exceptions.RequestException as error:
        print("\nNetwork/API request failed:")
        print(error)

    except Exception as error:
        print("\nDownload failed:")
        print(error)
