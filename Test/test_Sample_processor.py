from pathlib import Path

import torch

from src.RawDataLoader import RawDataLoader
from src.Group_Processor import Group_Processor
from src.Slice_Processor import Slice_Processor
from src.Sample_Processor import Sample_Processor
from src.BenchmarkData import BenchmarkData


# ==========================================================
# Configuration
# ==========================================================

ROOT_DIR = r"D:\Project\PHM_Data\RawData\CWRU"

GROUP_SIZE = 10000
SLICE_SIZE = 1024
SLICE_STRIDE = 512

TRAIN_CONDITIONS = [1797, 1797, 1750]
VALIDATION_CONDITIONS = [1797, 1772, 1730]
TEST_CONDITIONS = [1772, 1730]

TRAIN_SIZE = 2000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

EXCLUDED_FAULT_SIZES = ["028"]
OR_FAULT_POSITION = 6

RANDOM_SEED = 42

OUTPUT_DIR = (
    r"D:\Project\PHM_Data\SampledData"
    r"\10000-1024-512-ex028-10Lables"
)


# ==========================================================
# Utilities
# ==========================================================

def print_section(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def condition_requirements(conditions, total_size):
    """Calculate expected Condition allocation dynamically."""

    base = total_size // len(conditions)
    remainder = total_size % len(conditions)

    requirements = {}

    for index, condition in enumerate(conditions):

        amount = base + (
            1 if index < remainder else 0
        )

        requirements[condition] = (
            requirements.get(condition, 0)
            + amount
        )

    return requirements


# ==========================================================
# 1. Basic Sample Processor Output
# ==========================================================

def test_sample_counts(
    train_set,
    validation_set,
    test_set,
):
    print_section("Sample Count Test")

    print(f"Train:       {len(train_set)}")
    print(f"Validation:  {len(validation_set)}")
    print(f"Test:        {len(test_set)}")

    assert len(train_set) == TRAIN_SIZE
    assert len(validation_set) == VALIDATION_SIZE
    assert len(test_set) == TEST_SIZE

    print("\nPASS")


# ==========================================================
# 2. Condition Distribution
# ==========================================================

def get_condition_distribution(data_list):
    distribution = {}

    for data in data_list:
        rpm = data.metadata["info"]["rpm"]

        distribution[rpm] = (
            distribution.get(rpm, 0) + 1
        )

    return distribution


def test_condition_distribution(
    train_set,
    validation_set,
    test_set,
):
    print_section("Condition Distribution Test")

    expected_train = condition_requirements(
        TRAIN_CONDITIONS,
        TRAIN_SIZE,
    )

    expected_validation = condition_requirements(
        VALIDATION_CONDITIONS,
        VALIDATION_SIZE,
    )

    expected_test = condition_requirements(
        TEST_CONDITIONS,
        TEST_SIZE,
    )

    actual_train = get_condition_distribution(
        train_set
    )

    actual_validation = get_condition_distribution(
        validation_set
    )

    actual_test = get_condition_distribution(
        test_set
    )

    print("Train:")
    print(f"  Expected: {expected_train}")
    print(f"  Actual:   {actual_train}")

    print("\nValidation:")
    print(f"  Expected: {expected_validation}")
    print(f"  Actual:   {actual_validation}")

    print("\nTest:")
    print(f"  Expected: {expected_test}")
    print(f"  Actual:   {actual_test}")

    assert actual_train == expected_train
    assert actual_validation == expected_validation
    assert actual_test == expected_test

    print("\nPASS")


# ==========================================================
# 3. Sample ID
# ==========================================================

def test_sample_ids(
    train_set,
    validation_set,
    test_set,
):
    print_section("Sample ID Test")

    for data_list, expected_size in [
        (train_set, TRAIN_SIZE),
        (validation_set, VALIDATION_SIZE),
        (test_set, TEST_SIZE),
    ]:

        sample_ids = [
            data.metadata["sample_info"]["sample_id"]
            for data in data_list
        ]

        assert sample_ids == list(
            range(expected_size)
        )

    print("Train IDs:       0 ->", len(train_set) - 1)
    print("Validation IDs:  0 ->", len(validation_set) - 1)
    print("Test IDs:        0 ->", len(test_set) - 1)

    print("\nPASS")


# ==========================================================
# 4. Status
# ==========================================================

def test_status(
    train_set,
    validation_set,
    test_set,
):
    print_section("Status Test")

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    statuses = {
        data.metadata["status"]
        for data in all_data
    }

    print(f"Statuses: {statuses}")

    assert statuses == {"sampled"}

    print("\nPASS")


# ==========================================================
# 5. Label
# ==========================================================

def test_labels(
    train_set,
    validation_set,
    test_set,
):
    print_section("Label Test")

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    labels = {
        data.y
        for data in all_data
    }

    print(f"Unique labels: {len(labels)}")

    assert len(labels) > 0

    for data in all_data:

        info = data.metadata["info"]

        if info["fault_label"] == "Normal":

            expected = (
                f"{info['rpm']}-Normal"
            )

        else:

            expected = (
                f"{info['rpm']}-"
                f"{info['fault_size']}-"
                f"{info['fault_label']}"
            )

        assert data.y == expected

    print("\nPASS")


# ==========================================================
# 6. Excluded Fault Size
# ==========================================================

def test_excluded_fault_size(
    train_set,
    validation_set,
    test_set,
):
    print_section("Excluded Fault Size Test")

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    for data in all_data:

        fault_size = data.metadata["info"][
            "fault_size"
        ]

        assert fault_size not in (
            EXCLUDED_FAULT_SIZES
        )

    print(
        f"Excluded fault sizes: "
        f"{EXCLUDED_FAULT_SIZES}"
    )

    print("\nPASS")


# ==========================================================
# 7. OR Position
# ==========================================================

def test_or_position(
    train_set,
    validation_set,
    test_set,
):
    print_section("OR Fault Position Test")

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    or_data = [
        data
        for data in all_data
        if data.metadata["info"]["fault_label"]
        == "OR"
    ]

    positions = {
        data.metadata["extra_info"][
            "fault_position_num"
        ]
        for data in or_data
    }

    print(f"OR positions: {positions}")

    assert positions == {
        OR_FAULT_POSITION
    }

    print("\nPASS")


# ==========================================================
# 8. Metadata Integrity
# ==========================================================

def test_metadata(
    train_set,
    validation_set,
    test_set,
):
    print_section("Metadata Test")

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    required_keys = [
        "dataset",
        "info",
        "group_info",
        "extra_info",
        "sample_info",
        "status",
    ]

    for data in all_data:

        for key in required_keys:
            assert key in data.metadata

        assert (
            "sample_id"
            in data.metadata["sample_info"]
        )

    print("Required metadata fields exist.")

    print("\nPASS")


# ==========================================================
# 9. Group Leakage
# ==========================================================

def get_group_keys(data_list):
    keys = set()

    for data in data_list:

        metadata = data.metadata

        source = (
            metadata["dataset"],
            metadata["info"]["rpm"],
            metadata["info"]["load"],
            metadata["info"]["fault_size"],
            metadata["info"]["file_name"],
            metadata["extra_info"]["source_key"],
        )

        group_id = (
            metadata["group_info"]["group_id"]
        )

        keys.add(
            (
                source,
                group_id,
            )
        )

    return keys


def test_group_leakage(
    train_set,
    validation_set,
    test_set,
):
    print_section("Group Leakage Test")

    train_groups = get_group_keys(
        train_set
    )

    validation_groups = get_group_keys(
        validation_set
    )

    test_groups = get_group_keys(
        test_set
    )

    train_val = (
        train_groups & validation_groups
    )

    train_test = (
        train_groups & test_groups
    )

    validation_test = (
        validation_groups & test_groups
    )

    print(
        f"Train groups:       {len(train_groups)}"
    )

    print(
        f"Validation groups:  "
        f"{len(validation_groups)}"
    )

    print(
        f"Test groups:        {len(test_groups)}"
    )

    assert not train_val
    assert not train_test
    assert not validation_test

    print("\nPASS")


# ==========================================================
# 10. Global Shuffle
# ==========================================================

def test_global_shuffle(
    train_set,
    validation_set,
    test_set,
):
    print_section("Global Shuffle Preview")

    print("\nTrain:")
    for data in train_set[:10]:

        print(
            data.metadata["info"]["file_name"],
            "|",
            data.metadata["info"]["rpm"],
            "|",
            data.metadata["info"]["fault_label"],
            "|",
            data.metadata["group_info"]["group_id"],
        )

    print("\nValidation:")
    for data in validation_set[:10]:

        print(
            data.metadata["info"]["file_name"],
            "|",
            data.metadata["info"]["rpm"],
            "|",
            data.metadata["info"]["fault_label"],
            "|",
            data.metadata["group_info"]["group_id"],
        )

    print("\nTest:")
    for data in test_set[:10]:

        print(
            data.metadata["info"]["file_name"],
            "|",
            data.metadata["info"]["rpm"],
            "|",
            data.metadata["info"]["fault_label"],
            "|",
            data.metadata["group_info"]["group_id"],
        )

    print("\nPASS")


# ==========================================================
# 11. Saved File
# ==========================================================

def test_saved_file(
    sample_processor,
):
    print_section("Saved File Test")

    output_file = sample_processor.output_file

    print(f"Output file:")
    print(f"  {output_file}")

    assert output_file.exists()
    assert output_file.is_file()

    print("\nPASS")


# ==========================================================
# 12. Reload Saved File
# ==========================================================

def test_reload_saved_data(
    sample_processor,
):
    print_section("Reload Saved Data Test")

    saved_data = torch.load(
        sample_processor.output_file,
        weights_only=False,
    )

    assert isinstance(
        saved_data,
        dict,
    )

    assert "train" in saved_data
    assert "validation" in saved_data
    assert "test" in saved_data

    train_set = saved_data["train"]
    validation_set = saved_data["validation"]
    test_set = saved_data["test"]

    print(
        f"Reloaded Train:       "
        f"{len(train_set)}"
    )

    print(
        f"Reloaded Validation:  "
        f"{len(validation_set)}"
    )

    print(
        f"Reloaded Test:        "
        f"{len(test_set)}"
    )

    assert isinstance(
        train_set,
        list,
    )

    assert isinstance(
        validation_set,
        list,
    )

    assert isinstance(
        test_set,
        list,
    )

    assert len(train_set) == TRAIN_SIZE
    assert len(validation_set) == VALIDATION_SIZE
    assert len(test_set) == TEST_SIZE

    all_data = (
        train_set
        + validation_set
        + test_set
    )

    assert all(
        isinstance(
            data,
            BenchmarkData,
        )
        for data in all_data
    )

    print("\nPASS")


# ==========================================================
# Main
# ==========================================================

def main():

    print_section(
        "Sample Processor Full Test"
    )

    # ------------------------------------------------------
    # Raw Data
    # ------------------------------------------------------

    print_section("Raw Data Loading")

    raw_loader = RawDataLoader(
        ROOT_DIR
    )

    raw_data = raw_loader.load()

    print(
        f"Raw BenchmarkData: "
        f"{len(raw_data)}"
    )

    # ------------------------------------------------------
    # Group Processor
    # ------------------------------------------------------

    print_section("Group Processing")

    group_processor = Group_Processor(
        group_size=GROUP_SIZE
    )

    grouped_data = group_processor.process(
        raw_data
    )

    print(
        f"Grouped BenchmarkData: "
        f"{len(grouped_data)}"
    )

    # ------------------------------------------------------
    # Slice Processor
    # ------------------------------------------------------

    print_section("Slice Processing")

    slice_processor = Slice_Processor(
        slice_size=SLICE_SIZE,
        stride=SLICE_STRIDE,
    )

    sliced_data = slice_processor.process(
        grouped_data
    )

    print(
        f"Sliced BenchmarkData: "
        f"{len(sliced_data)}"
    )

    # ------------------------------------------------------
    # Sample Processor
    # ------------------------------------------------------

    print_section("Sample Processing")

    sample_processor = Sample_Processor(
        train_conditions=TRAIN_CONDITIONS,
        validation_conditions=VALIDATION_CONDITIONS,
        test_conditions=TEST_CONDITIONS,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        excluded_fault_sizes=EXCLUDED_FAULT_SIZES,
        or_fault_position=OR_FAULT_POSITION,
        random_seed=RANDOM_SEED,
        output_dir=OUTPUT_DIR,
    )

    train_set, validation_set, test_set = (
        sample_processor.process(
            sliced_data
        )
    )

    print(
        f"Train:       {len(train_set)}"
    )

    print(
        f"Validation:  {len(validation_set)}"
    )

    print(
        f"Test:        {len(test_set)}"
    )

    # ------------------------------------------------------
    # Tests
    # ------------------------------------------------------

    test_sample_counts(
        train_set,
        validation_set,
        test_set,
    )

    test_condition_distribution(
        train_set,
        validation_set,
        test_set,
    )

    test_sample_ids(
        train_set,
        validation_set,
        test_set,
    )

    test_status(
        train_set,
        validation_set,
        test_set,
    )

    test_labels(
        train_set,
        validation_set,
        test_set,
    )

    test_excluded_fault_size(
        train_set,
        validation_set,
        test_set,
    )

    test_or_position(
        train_set,
        validation_set,
        test_set,
    )

    test_metadata(
        train_set,
        validation_set,
        test_set,
    )

    test_group_leakage(
        train_set,
        validation_set,
        test_set,
    )

    test_global_shuffle(
        train_set,
        validation_set,
        test_set,
    )

    # ------------------------------------------------------
    # Saved File Tests
    # ------------------------------------------------------

    test_saved_file(
        sample_processor
    )

    test_reload_saved_data(
        sample_processor
    )

    # ------------------------------------------------------
    # Complete
    # ------------------------------------------------------

    print_section(
        "ALL SAMPLE PROCESSOR TESTS PASSED"
    )


if __name__ == "__main__":
    main()