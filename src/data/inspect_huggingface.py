
from huggingface_hub import HfApi

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"


def inspect_dataset():
    print("\nMOSAIC 2.0 - HUGGING FACE DATASET INSPECTION")
    print("=" * 55)

    api = HfApi()

    print("\nChecking dataset repository...")

    try:
        info = api.dataset_info(DATASET_ID)

        print("Dataset found successfully!")
        print("Dataset ID:", info.id)

        print("\nAvailable repository files:")

        files = api.list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset"
        )

        for file in files:
            print("-", file)

        print("\nInspection completed.")

    except Exception as error:
        print("\nDataset inspection failed!")
        print("Error:", error)


if __name__ == "__main__":
    inspect_dataset()
