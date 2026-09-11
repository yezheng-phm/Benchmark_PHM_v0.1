from src.loaders.CWRULoader import CWRULoader
from src.Group_Processor import Group_Processor


# CWRU dataset root directory.
root_dir = r"D:\Project\PHM_Data\RawData\CWRU"


# --------------------------------------------------
# Load raw CWRU data.
# --------------------------------------------------
loader = CWRULoader(root_dir)
raw_data_list = loader.load()


# --------------------------------------------------
# Process raw data into fixed-length groups.
# --------------------------------------------------
group_processor = Group_Processor(
    group_size=10000
)

grouped_data_list = group_processor.process(
    raw_data_list
)


print("=" * 70)
print("Group Processor Test")
print("=" * 70)

print(f"\nRaw BenchmarkData:   {len(raw_data_list)}")
print(f"Group BenchmarkData: {len(grouped_data_list)}")


# --------------------------------------------------
# Display all Group BenchmarkData objects.
# --------------------------------------------------
for index, data in enumerate(
    grouped_data_list,
    start=1
):

    metadata = data.metadata

    print("\n" + "=" * 70)
    print(
        f"Group BenchmarkData "
        f"[{index}/{len(grouped_data_list)}]"
    )
    print("-" * 70)

    # X information.
    print("X shape:")
    print(f"  {data.X.shape}")

    # Label information.
    print("\ny:")
    print(f"  {data.y}")

    # Metadata.
    print("\nmetadata:")

    print("  status:")
    print(f"    {metadata['status']}")

    print("\n  dataset:")
    print(f"    {metadata['dataset']}")

    print("\n  info:")
    for key, value in metadata["info"].items():
        print(f"    {key}: {value}")

    print("\n  extra_info:")
    for key, value in metadata["extra_info"].items():
        print(f"    {key}: {value}")

    print("\n  group_info:")
    for key, value in metadata["group_info"].items():
        print(f"    {key}: {value}")

    print("\n  slice_info:")
    for key, value in metadata["slice_info"].items():
        print(f"    {key}: {value}")

    print("\n  sample_info:")
    for key, value in metadata["sample_info"].items():
        print(f"    {key}: {value}")


print("\n" + "=" * 70)
print("Test finished.")
print("=" * 70)