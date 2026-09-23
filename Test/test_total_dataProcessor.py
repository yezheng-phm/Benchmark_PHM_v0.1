from DataProcessPart.Loaders.CWRU_Loader_new import CWRU_Loader
from DataProcessPart.Group_Processor import Group_Processor
from DataProcessPart.Slice_Processor import Slice_Processor
from DataProcessPart.Sample_Processor import Sample_Processor


# -----------------------------------------------------------------------
# CWRU data configuration
# -----------------------------------------------------------------------

RAW_DATA_ROOT_PATH = r"D:\Project\PHM_Data\RawData"
DATA_SET_NAME = "CWRU"
CURRENT_DATA_PATH = r"CWRU"


# -----------------------------------------------------------------------
# Total DataProcessPart test
# -----------------------------------------------------------------------

def main():

    print("=" * 70)
    print("Benchmark PHM DataProcessPart - Total Pipeline Test")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 1. CWRU Loader
    # -------------------------------------------------------------------

    print("\n[1] CWRU Loader")

    loader = CWRU_Loader(
        RAW_DATA_ROOT_PATH,
        DATA_SET_NAME,
        CURRENT_DATA_PATH,
    )

    loaded_data = loader.run()

    print(f"Number of containers: {len(loaded_data)}")

    if loaded_data:
        print(
            f"First X_data shape: "
            f"{loaded_data[0].X_data.shape}"
        )
        print(
            f"First y: "
            f"{loaded_data[0].y}"
        )


    # -------------------------------------------------------------------
    # 2. Group Processor
    # -------------------------------------------------------------------

    print("\n[2] Group Processor")

    group_processor = Group_Processor(
        loaded_data
    )

    grouped_data = group_processor.run()

    print(f"Number of groups: {len(grouped_data)}")

    if grouped_data:
        print(
            f"First group X_data shape: "
            f"{grouped_data[0].X_data.shape}"
        )
        print(
            f"First group info: "
            f"{grouped_data[0].group_info}"
        )


    # -------------------------------------------------------------------
    # 3. Slice Processor
    # -------------------------------------------------------------------

    print("\n[3] Slice Processor")

    slice_processor = Slice_Processor(
        grouped_data
    )

    sliced_data = slice_processor.run()

    print(f"Number of slices: {len(sliced_data)}")

    if sliced_data:
        print(
            f"First slice X_data shape: "
            f"{sliced_data[0].X_data.shape}"
        )
        print(
            f"First slice info: "
            f"{sliced_data[0].slice_info}"
        )


    # -------------------------------------------------------------------
    # 4. Sample Processor
    # -------------------------------------------------------------------

    print("\n[4] Sample Processor")

    sample_processor = Sample_Processor(
        sliced_data
    )

    sample_data = sample_processor.run()

    print(
        f"Training samples: "
        f"{len(sample_data['training'])}"
    )

    print(
        f"Validation samples: "
        f"{len(sample_data['validation'])}"
    )

    print(
        f"Test samples: "
        f"{len(sample_data['test'])}"
    )

    print(
        f"Saved file: "
        f"{sample_processor.sample_data_save_path}\\"
        f"{sample_processor.sample_data_save_name}"
    )


    # -------------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Total pipeline test completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()