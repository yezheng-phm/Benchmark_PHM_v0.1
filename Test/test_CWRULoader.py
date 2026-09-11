from src.loaders.CWRULoader import CWRULoader


# CWRU dataset root directory.
root_dir = r"D:\Project\PHM_Data\RawData\CWRU"


# Initialize the loader.
loader = CWRULoader(root_dir)


# Load all CWRU data.
data_list = loader.load()


print("=" * 70)
print("CWRU Loader Test")
print("=" * 70)

print(f"\nTotal files loaded: {len(data_list)}")


# --------------------------------------------------
# Display all loaded BenchmarkData objects.
# --------------------------------------------------

for index, data in enumerate(data_list, start=1):

    metadata = data.metadata

    print("\n" + "=" * 70)
    print(f"BenchmarkData [{index}/{len(data_list)}]")
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