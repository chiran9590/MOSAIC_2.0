
import requests

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"

API_URL = "https://datasets-server.huggingface.co/parquet"


def check_metadata():

    print("\nMOSAIC 2.0 - PARQUET METADATA")
    print("=" * 55)

    try:
        response = requests.get(
            API_URL,
            params={"dataset": DATASET_ID},
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        print("\nParquet files available:\n")

        for file in result.get("parquet_files", []):

            if file.get("split") == "minute":

                print("Filename:", file.get("filename"))
                print("Size:", round(file.get("size", 0) / (1024**3), 3), "GB")
                print("URL:", file.get("url"))
                print("-" * 50)

        print("\nPartial:", result.get("partial"))

    except Exception as error:
        print("\nMetadata request failed:", error)


if __name__ == "__main__":
    check_metadata()
