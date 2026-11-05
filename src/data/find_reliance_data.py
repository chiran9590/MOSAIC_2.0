
import requests

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"

API_URL = "https://datasets-server.huggingface.co/filter"


def find_reliance():

    print("\nMOSAIC 2.0 - RELIANCE DATA SEARCH")
    print("=" * 55)

    params = {
        "dataset": DATASET_ID,
        "config": "default",
        "split": "minute",
        "where": '"symbol"=\'RELIANCE\'',
        "offset": 0,
        "length": 5
    }

    try:
        print("\nSearching for RELIANCE...")

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        print("\nAPI response received!")

        print("Total matching rows:",
              result.get("num_rows_total", "Unknown"))

        print("Partial indexing:",
              result.get("partial", "Not specified"))

        print("\nFirst 5 RELIANCE records:\n")

        rows = result.get("rows", [])

        for item in rows:
            row = item["row"]

            for column, value in row.items():
                print(f"{column}: {value}")

            print("-" * 40)

        if not rows:
            print("No records returned.")

        print("\nSearch completed.")

    except requests.exceptions.RequestException as error:
        print("\nAPI request failed:", error)

    except Exception as error:
        print("\nUnexpected error:", error)


if __name__ == "__main__":
    find_reliance()
