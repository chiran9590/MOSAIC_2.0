
import requests
import pyarrow.parquet as pq


DATASET_BASE_URL = (
    "https://huggingface.co/datasets/"
    "xxparthparekhxx/indian-stock-market-minute-data/"
    "resolve/refs%2Fconvert%2Fparquet/default/minute/"
)

TARGET_SYMBOL = "RELIANCE"


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
            raise RuntimeError(
                "Server does not support HTTP range requests."
            )

        content_range = response.headers.get("Content-Range")

        if not content_range:
            raise RuntimeError("Content-Range header is missing.")

        self.size = int(content_range.split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):

        if self.closed:
            raise ValueError("I/O operation on closed file")

        if whence == 0:
            self.position = offset

        elif whence == 1:
            self.position += offset

        elif whence == 2:
            self.position = self.size + offset

        else:
            raise ValueError("Invalid whence value")

        if self.position < 0:
            raise ValueError("Negative seek position")

        return self.position

    def read(self, size=-1):

        if self.closed:
            raise ValueError("I/O operation on closed file")

        if size is None or size < 0:
            size = self.size - self.position

        if size == 0 or self.position >= self.size:
            return b""

        end = min(self.position + size, self.size) - 1

        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}"
            },
            timeout=60
        )

        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                "Server did not return partial content."
            )

        data = response.content
        self.position += len(data)

        return data

    def flush(self):
        pass

    def close(self):

        if not self.closed:
            self.closed = True
            self.session.close()


def inspect_shard(shard_number):

    filename = f"{shard_number:04d}.parquet"
    url = DATASET_BASE_URL + filename

    print(f"\nInspecting shard {shard_number}: {filename}")

    reader = HTTPRangeReader(url)

    try:

        parquet_file = pq.ParquetFile(reader)
        metadata = parquet_file.metadata

        print("Total rows:", metadata.num_rows)
        print("Total row groups:", metadata.num_row_groups)

        schema = parquet_file.schema_arrow

        symbol_index = schema.get_field_index("symbol")

        if symbol_index == -1:
            print("Symbol column not found.")
            return

        possible_matches = 0

        for group_number in range(metadata.num_row_groups):

            group = metadata.row_group(group_number)
            column = group.column(symbol_index)
            stats = column.statistics

            if stats is None or not stats.has_min_max:
                continue

            minimum = stats.min
            maximum = stats.max

            if isinstance(minimum, bytes):
                minimum = minimum.decode("utf-8")

            if isinstance(maximum, bytes):
                maximum = maximum.decode("utf-8")

            if minimum <= TARGET_SYMBOL <= maximum:

                possible_matches += 1

                print(
                    f"POSSIBLE MATCH: Row group {group_number}"
                )

                print("Minimum symbol:", minimum)
                print("Maximum symbol:", maximum)

        if possible_matches == 0:
            print("No matching row groups found.")

    finally:
        reader.close()


def main():

    print("\nMOSAIC 2.0 - PARQUET METADATA SCAN")
    print("=" * 55)

    for shard_number in range(8):

        try:
            inspect_shard(shard_number)

        except Exception as error:
            print(
                f"Error in shard {shard_number}: "
                f"{type(error).__name__}: {error}"
            )

    print("\nMetadata scan completed.")


if __name__ == "__main__":
    main()
