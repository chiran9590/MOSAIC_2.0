
from datasets import load_dataset
from huggingface_hub import hf_hub_url

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"

PARQUET_FILES = [
    f"minute/train-{i:05d}.parquet"
    for i in range(8)
]


def inspect_minute_data():

    print("\nMOSAIC 2.0 - MINUTE DATA INSPECTION")
    print("=" * 55)

    urls = [
        hf_hub_url(
            repo_id=DATASET_ID,
            filename=file,
            repo_type="dataset"
        )
        for file in PARQUET_FILES
    ]

    try:
        print("\nConnecting to minute dataset...")

        dataset = load_dataset(
            "parquet",
            data_files={"train": urls},
            split="train",
            streaming=True
        )

        print("\nDataset connected successfully!")

        print("\nReading first 5 records...\n")

        for index, row in enumerate(dataset.take(5), start=1):

            print(f"RECORD {index}")
            print("-" * 40)

            for column, value in row.items():
                print(f"{column}: {value}")

            print()

        print("Sample inspection completed.")

    except Exception as error:
        print("\nInspection failed!")
        print("Error:", error)


if __name__ == "__main__":
    inspect_minute_data()
