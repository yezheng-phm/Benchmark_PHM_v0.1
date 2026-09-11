from src.RawDataLoader import RawDataLoader
from src.loaders.CWRULoader import CWRULoader


# CWRU dataset root directory.
root_dir = r"D:\Project\PHM_Data\RawData\CWRU"


# Initialize the dataset-specific loader.
cwru_loader = CWRULoader(root_dir)


# Pass the CWRU loader to the unified raw data loader.
raw_loader = RawDataLoader(cwru_loader)


# Load all raw data.
data_list = raw_loader.load()


print("=" * 70)
print("RawDataLoader Test")
print("=" * 70)

print(f"\nTotal files loaded: {len(data_list)}")


# Display the first five BenchmarkData objects.
for index, data in enumerate(data_list[:5], start=1):

    metadata = data.metadata

    print("\n" + "=" * 70)
    print(f"BenchmarkData [{index}/5]")
    print("-" * 70)

    print("X shape:")
    print(f"  {data.X.shape}")

    print("\ny:")
    print(f"  {data.y}")

    print("\nmetadata:")

    print("  dataset:")
    print(f"    {metadata['dataset']}")

    print("\n  info:")
    for key, value in metadata["info"].items():
        print(f"    {key}: {value}")

    print("\n  extra_info:")
    for key, value in metadata["extra_info"].items():
        print(f"    {key}: {value}")


print("\n" + "=" * 70)
print("Test finished.")
print("=" * 70)