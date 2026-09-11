from pathlib import Path

from scipy.io import loadmat


# CWRU dataset root directory
root_dir = Path(
    r"D:\Project\PHM_Data\RawData\CWRU"
)

# Find all .mat files
mat_files = list(root_dir.rglob("*.mat"))

print(f"Total .mat files: {len(mat_files)}")
print("=" * 80)


for i, file_path in enumerate(mat_files, start=1):

    print(f"\n[{i}/{len(mat_files)}]")
    print(f"File: {file_path.name}")
    print(f"Path: {file_path}")

    try:
        mat_data = loadmat(file_path)

        for key, value in mat_data.items():

            # Skip MATLAB internal fields
            if key.startswith("__"):
                continue

            print(f"\n  Key: {key}")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")

            # Print scalar value
            if value.size == 1:
                print(f"  Value: {value.squeeze()}")

    except Exception as e:
        print(f"\n  ERROR: {e}")

    print("\n" + "-" * 80)