
import requests
import pyarrow.parquet as pq
import pandas as pd
from io import BytesIO
from pathlib import Path


URL = (
    "https://huggingface.co/datasets/"
    "xxparthparekhxx/indian-stock-market-minute-data/"
    "resolve/refs%2Fconvert%2Fparquet/default/minute/0005.parquet"
)

TARGET_SYMBOL = "RELIANCE"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "reliance_1min.csv"

CHUNK_SIZE = 8 * 1024 * 1024


class HTTPRangeReader:

    def __init__(self, url):

        self.url = url
        self.position = 0
        self.closed = False
        self.session = requests.Session()

        response = self.session.get(
            url,
            headers={"Range": "bytes=0-0"},
            timeout=30
        )

        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError("HTTP range requests unavailable.")

        self.size = int(
            response.headers["Content-Range"].split("/")[-1]
        )

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):

        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError("Invalid whence")

        return self.position

    def read(self, size=-1):

        if self.closed:
            raise ValueError("Reader is closed")

        if size is None or size < 0:
            size = self.size - self.position

        if size <= 0 or self.position >= self.size:
            return b""

        end = min(self.position + size, self.size) - 1

        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}"
            },
            timeout=120
        )

        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError("Partial download failed")

        data = response.content
        self.position += len(data)

        return data

    def flush(self):
        pass

    def close(self):

        if not self.closed:
            self.closed = True
            self.session.close()


def main():

    print("\nMOSAIC 2.0 - RELIANCE DATA EXTRACTION")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = HTTPRangeReader(URL)

    try:

        parquet_file = pq.ParquetFile(reader)

        all_frames = []

        # Only inspect row groups 37 and 38.
        for group_number in [37, 38]:

            print(f"\nReading row group {group_number}...")

            table = parquet_file.read_row_group(
                group_number,
                columns=[
                    "symbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            )

            df = table.to_pandas()

            reliance_df = df[
                df["symbol"] == TARGET_SYMBOL
            ].copy()

            print(
                "RELIANCE records found:",
                len(reliance_df)
            )

            if not reliance_df.empty:
                all_frames.append(reliance_df)

        if not all_frames:
            print("\nNo RELIANCE records found.")
            return

        result = pd.concat(
            all_frames,
            ignore_index=True
        )

        result = result.drop(columns=["symbol"])

        # Convert UTC timestamps to IST.
        result["timestamp"] = (
            pd.to_datetime(result["timestamp"], utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )

        result = result.sort_values("timestamp")

        result = result.drop_duplicates(
            subset=["timestamp"]
        )

        result.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nEXTRACTION COMPLETED")
        print("=" * 55)

        print("Total records:", len(result))
        print("Start:", result["timestamp"].min())
        print("End:", result["timestamp"].max())
        print("Saved to:", OUTPUT_FILE)

        print("\nFirst five records:")
        print(result.head())

    finally:
        reader.close()


if __name__ == "__main__":
    main()
