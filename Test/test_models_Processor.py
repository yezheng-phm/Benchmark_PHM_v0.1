#----------------------------------------------------------------------
#--test SampleProcessor
#--CWRULoader -> Group_Processor -> Slice_Processor
#--are only used to prepare input data for SampleProcessor.
#----------------------------------------------------------------------


from DataProcessPart.DataProcessContainer import DataProcessContainer
from DataProcessPart.Loaders.CWRU_Loader_new import CWRULoader
from DataProcessPart.Group_Processor import Group_Processor
from DataProcessPart.Slice_Processor import Slice_Processor
from DataProcessPart.Sample_Processor import SampleProcessor


#-------------------set test data path-------------------

RAW_DATA_ROOT_PATH = r"D:\Project\PHM_Data\RawData"
DATA_SET_NAME = "CWRU"
CURRENT_DATA_PATH = r"CWRU"


#======================================================================
#--Prepare data for SampleProcessor
#======================================================================

loader = CWRULoader(
    raw_data_root_path=RAW_DATA_ROOT_PATH,
    data_set_name=DATA_SET_NAME,
    current_data_path=CURRENT_DATA_PATH,
)

data_list = loader.run()


group_processor = Group_Processor(
    data_list=data_list,
)

grouped_data = group_processor.run()


slice_processor = Slice_Processor(
    data_list=grouped_data,
)

sliced_data = slice_processor.run()


#======================================================================
#--SampleProcessor Test
#======================================================================

sample_processor = SampleProcessor(
    data=sliced_data,
)

print("\n========== SampleProcessor Test ==========\n")


#----------------------------------------------------------------------
#--Step 1: test _filter_data()
#----------------------------------------------------------------------

filtered_data = sample_processor._filter_data()

print("Filtered data:")
print(f"Input slices: {len(sliced_data)}")
print(f"Filtered slices: {len(filtered_data)}")


#-------------------validate filtering result-------------------

assert len(filtered_data) > 0
assert len(filtered_data) <= len(sliced_data)

for data in filtered_data:

    metadata_info = data.metadata["metadata_info"]

    fault_size = metadata_info["fault_size"]
    fault_label = metadata_info["fault_label"]
    fault_position_num = metadata_info["fault_position_num"]

    # Excluded fault sizes must not appear.
    assert (
        fault_size
        not in sample_processor.excluded_fault_sizes
    )

    # Only selected OR fault positions are allowed.
    if fault_label == "OR":

        assert (
            fault_position_num
            in sample_processor.or_fault_positions
        )


print("[PASS] SampleProcessor _filter_data() validation.")


#----------------------------------------------------------------------
#--Step 2: test _build_data_index()
#----------------------------------------------------------------------

data_index = sample_processor._build_data_index(
    filtered_data
)

print("\nData index structure:")
print("Domain -> File -> Group -> Slice")

print(
    f"Indexed domains: "
    f"{list(data_index.keys())}"
)


#-------------------validate data index-------------------

assert isinstance(data_index, dict)
assert len(data_index) > 0

for domain, file_dict in data_index.items():

    assert isinstance(domain, int)
    assert isinstance(file_dict, dict)
    assert len(file_dict) > 0

    for file_name, group_dict in file_dict.items():

        assert isinstance(file_name, str)
        assert isinstance(group_dict, dict)
        assert len(group_dict) > 0

        for group_id, slice_ids in group_dict.items():

            assert isinstance(group_id, int)
            assert isinstance(slice_ids, list)
            assert len(slice_ids) > 0

            for slice_id in slice_ids:

                assert isinstance(slice_id, int)


print("[PASS] SampleProcessor _build_data_index() validation.")


#----------------------------------------------------------------------
#--Step 3: test _calculate_file_sample_requirements()
#----------------------------------------------------------------------

file_sample_requirements = (
    sample_processor._calculate_file_sample_requirements(
        data_index
    )
)


print("\nFile sample requirements:")
print(
    "Domain -> File -> "
    "Training / Validation / Test"
)


for domain, file_dict in file_sample_requirements.items():

    print(f"\nDomain: {domain}")

    for file_name, split_requirements in file_dict.items():

        print(
            f"  {file_name}: "
            f"training={split_requirements['training']}, "
            f"validation={split_requirements['validation']}, "
            f"test={split_requirements['test']}"
        )


#-------------------validate file requirements-------------------

assert isinstance(
    file_sample_requirements,
    dict,
)

assert len(file_sample_requirements) > 0


for domain, file_dict in file_sample_requirements.items():

    assert domain in data_index

    assert isinstance(file_dict, dict)
    assert len(file_dict) > 0

    for file_name, split_requirements in file_dict.items():

        # File must exist in filtered data index.
        assert file_name in data_index[domain]

        assert "training" in split_requirements
        assert "validation" in split_requirements
        assert "test" in split_requirements

        assert isinstance(
            split_requirements["training"],
            int,
        )

        assert isinstance(
            split_requirements["validation"],
            int,
        )

        assert isinstance(
            split_requirements["test"],
            int,
        )

        assert split_requirements["training"] >= 0
        assert split_requirements["validation"] >= 0
        assert split_requirements["test"] >= 0


#----------------------------------------------------------------------
#--Validate total sample requirements
#----------------------------------------------------------------------

training_total = sum(
    split_requirements["training"]
    for file_dict in file_sample_requirements.values()
    for split_requirements in file_dict.values()
)

validation_total = sum(
    split_requirements["validation"]
    for file_dict in file_sample_requirements.values()
    for split_requirements in file_dict.values()
)

test_total = sum(
    split_requirements["test"]
    for file_dict in file_sample_requirements.values()
    for split_requirements in file_dict.values()
)


print("\nTotal sample requirements:")
print(f"Training:   {training_total}")
print(f"Validation: {validation_total}")
print(f"Test:       {test_total}")


assert training_total == sample_processor.train_size
assert validation_total == sample_processor.validation_size
assert test_total == sample_processor.test_size


print(
    "[PASS] "
    "SampleProcessor "
    "_calculate_file_sample_requirements() validation."
)


#----------------------------------------------------------------------
#--Validate that requirements only use configured domains
#----------------------------------------------------------------------

configured_domains = set(
    sample_processor.train_domain
    + sample_processor.validation_domain
    + sample_processor.test_domain
)


for domain in file_sample_requirements:

    assert domain in configured_domains


print("[PASS] SampleProcessor domain validation.")


#======================================================================
#--Step 4: validate Group allocation feasibility
#--
#--This test does NOT generate final Samples.
#--It validates whether the current quota requirements can be
#--satisfied by allocating complete Groups without cross-split leakage.
#======================================================================

print(
    "\n========== "
    "SampleProcessor Group Allocation Feasibility Test "
    "==========\n"
)


#----------------------------------------------------------------------
#--Helper: calculate number of slices available in a group
#----------------------------------------------------------------------

def _get_group_slice_count(
    domain,
    file_name,
    group_id,
):
    return len(
        data_index[
            domain
        ][
            file_name
        ][
            group_id
        ]
    )


#----------------------------------------------------------------------
#--Validate every File independently
#----------------------------------------------------------------------

for domain, file_dict in file_sample_requirements.items():

    for file_name, split_requirements in file_dict.items():

        group_dict = data_index[domain][file_name]

        group_ids = list(group_dict.keys())

        assert len(group_ids) > 0


        #--------------------------------------------------------------
        #--File-level requirements
        #--------------------------------------------------------------

        training_required = (
            split_requirements["training"]
        )

        validation_required = (
            split_requirements["validation"]
        )

        test_required = (
            split_requirements["test"]
        )


        total_required = (
            training_required
            + validation_required
            + test_required
        )


        #--------------------------------------------------------------
        #--Each Group is indivisible.
        #--Calculate the number of Groups needed sequentially.
        #--------------------------------------------------------------

        remaining_groups = list(group_ids)

        training_groups = []
        validation_groups = []
        test_groups = []


        #--------------------------------------------------------------
        #--Training
        #--------------------------------------------------------------

        training_capacity = 0

        while (
            training_capacity < training_required
            and remaining_groups
        ):

            group_id = remaining_groups.pop(0)

            training_groups.append(group_id)

            training_capacity += (
                _get_group_slice_count(
                    domain,
                    file_name,
                    group_id,
                )
            )


        #--------------------------------------------------------------
        #--Validation
        #--------------------------------------------------------------

        validation_capacity = 0

        while (
            validation_capacity < validation_required
            and remaining_groups
        ):

            group_id = remaining_groups.pop(0)

            validation_groups.append(group_id)

            validation_capacity += (
                _get_group_slice_count(
                    domain,
                    file_name,
                    group_id,
                )
            )


        #--------------------------------------------------------------
        #--Test
        #--------------------------------------------------------------

        test_capacity = 0

        while (
            test_capacity < test_required
            and remaining_groups
        ):

            group_id = remaining_groups.pop(0)

            test_groups.append(group_id)

            test_capacity += (
                _get_group_slice_count(
                    domain,
                    file_name,
                    group_id,
                )
            )


        #--------------------------------------------------------------
        #--Validate capacity
        #--------------------------------------------------------------

        if training_capacity < training_required:

            raise ValueError(
                f"Insufficient Group capacity for "
                f"Training: "
                f"domain={domain}, "
                f"file={file_name}, "
                f"required={training_required}, "
                f"available={training_capacity}"
            )


        if validation_capacity < validation_required:

            raise ValueError(
                f"Insufficient Group capacity for "
                f"Validation: "
                f"domain={domain}, "
                f"file={file_name}, "
                f"required={validation_required}, "
                f"available={validation_capacity}"
            )


        if test_capacity < test_required:

            raise ValueError(
                f"Insufficient Group capacity for "
                f"Test: "
                f"domain={domain}, "
                f"file={file_name}, "
                f"required={test_required}, "
                f"available={test_capacity}"
            )


        #--------------------------------------------------------------
        #--Validate Group isolation
        #--------------------------------------------------------------

        training_group_set = set(
            training_groups
        )

        validation_group_set = set(
            validation_groups
        )

        test_group_set = set(
            test_groups
        )


        assert training_group_set.isdisjoint(
            validation_group_set
        )

        assert training_group_set.isdisjoint(
            test_group_set
        )

        assert validation_group_set.isdisjoint(
            test_group_set
        )


        #--------------------------------------------------------------
        #--Print allocation information
        #--------------------------------------------------------------

        print(
            f"Domain={domain}, "
            f"File={file_name}"
        )

        print(
            f"  Groups available: "
            f"{len(group_ids)}"
        )

        print(
            f"  Training: "
            f"required={training_required}, "
            f"groups={len(training_groups)}, "
            f"capacity={training_capacity}"
        )

        print(
            f"  Validation: "
            f"required={validation_required}, "
            f"groups={len(validation_groups)}, "
            f"capacity={validation_capacity}"
        )

        print(
            f"  Test: "
            f"required={test_required}, "
            f"groups={len(test_groups)}, "
            f"capacity={test_capacity}"
        )


print(
    "\n[PASS] "
    "SampleProcessor Group allocation feasibility validation."
)


#======================================================================
#--Step 5: test _allocate_groups()
#======================================================================

print(
    "\n========== "
    "SampleProcessor Group Allocation Test "
    "==========\n"
)


group_allocation = sample_processor._allocate_groups(
    data_index=data_index,
    file_sample_requirements=file_sample_requirements,
)


#----------------------------------------------------------------------
#--Validate group allocation structure
#----------------------------------------------------------------------

assert isinstance(group_allocation, dict)
assert len(group_allocation) > 0


for domain, file_dict in group_allocation.items():

    assert domain in file_sample_requirements
    assert isinstance(file_dict, dict)

    for file_name, split_groups in file_dict.items():

        assert file_name in file_sample_requirements[domain]

        assert isinstance(split_groups, dict)

        assert "training" in split_groups
        assert "validation" in split_groups
        assert "test" in split_groups

        assert isinstance(
            split_groups["training"],
            list,
        )

        assert isinstance(
            split_groups["validation"],
            list,
        )

        assert isinstance(
            split_groups["test"],
            list,
        )


#----------------------------------------------------------------------
#--Validate Group isolation and capacity
#----------------------------------------------------------------------

for domain, file_dict in file_sample_requirements.items():

    for file_name, split_requirements in file_dict.items():

        group_dict = data_index[domain][file_name]

        allocated_groups = group_allocation[
            domain
        ][
            file_name
        ]

        training_groups = allocated_groups["training"]
        validation_groups = allocated_groups["validation"]
        test_groups = allocated_groups["test"]

        training_group_set = set(training_groups)
        validation_group_set = set(validation_groups)
        test_group_set = set(test_groups)

        #--------------------------------------------------------------
        #--No Group can appear in more than one split.
        #--------------------------------------------------------------

        assert training_group_set.isdisjoint(
            validation_group_set
        )

        assert training_group_set.isdisjoint(
            test_group_set
        )

        assert validation_group_set.isdisjoint(
            test_group_set
        )

        #--------------------------------------------------------------
        #--Every allocated Group must exist in the data index.
        #--------------------------------------------------------------

        for group_id in (
            training_groups
            + validation_groups
            + test_groups
        ):

            assert group_id in group_dict

        #--------------------------------------------------------------
        #--Calculate Group capacities.
        #--------------------------------------------------------------

        training_capacity = sum(
            len(group_dict[group_id])
            for group_id in training_groups
        )

        validation_capacity = sum(
            len(group_dict[group_id])
            for group_id in validation_groups
        )

        test_capacity = sum(
            len(group_dict[group_id])
            for group_id in test_groups
        )

        #--------------------------------------------------------------
        #--Each split must have enough Slice capacity.
        #--------------------------------------------------------------

        assert (
            training_capacity
            >= split_requirements["training"]
        )

        assert (
            validation_capacity
            >= split_requirements["validation"]
        )

        assert (
            test_capacity
            >= split_requirements["test"]
        )

        print(
            f"Domain={domain}, "
            f"File={file_name}"
        )

        print(
            f"  Training: "
            f"required={split_requirements['training']}, "
            f"groups={len(training_groups)}, "
            f"capacity={training_capacity}"
        )

        print(
            f"  Validation: "
            f"required={split_requirements['validation']}, "
            f"groups={len(validation_groups)}, "
            f"capacity={validation_capacity}"
        )

        print(
            f"  Test: "
            f"required={split_requirements['test']}, "
            f"groups={len(test_groups)}, "
            f"capacity={test_capacity}"
        )


print(
    "\n[PASS] "
    "SampleProcessor _allocate_groups() validation."
)


# ============================================================
# Step 6: Test _select_slices()
# ============================================================

slice_selection = sample_processor._select_slices(
    data_index=data_index,
    file_sample_requirements=file_sample_requirements,
    group_allocation=group_allocation,
)

# ------------------------------------------------------------
# 6.1 Check top-level structure
# ------------------------------------------------------------

assert set(slice_selection.keys()) == set(
    file_sample_requirements.keys()
)

# ------------------------------------------------------------
# 6.2 Check domain / file / split structure
# ------------------------------------------------------------

for domain, file_requirements in file_sample_requirements.items():

    assert domain in slice_selection

    for file_name, split_requirements in file_requirements.items():

        assert file_name in slice_selection[domain]

        split_selection = slice_selection[domain][file_name]

        assert set(split_selection.keys()) == {
            "training",
            "validation",
            "test",
        }

        # ----------------------------------------------------
        # 6.3 Check each split exact number of selected slices
        # ----------------------------------------------------

        for split_name in (
            "training",
            "validation",
            "test",
        ):

            selected_slices = split_selection[split_name]

            required_samples = split_requirements[split_name]

            assert len(selected_slices) == required_samples, (
                f"Slice selection size mismatch: "
                f"domain={domain}, "
                f"file={file_name}, "
                f"split={split_name}, "
                f"expected={required_samples}, "
                f"actual={len(selected_slices)}"
            )

            # ------------------------------------------------
            # 6.4 Each selected item must be (group_id, slice_id)
            # ------------------------------------------------

            for selected_item in selected_slices:

                assert isinstance(selected_item, tuple)
                assert len(selected_item) == 2

                group_id, slice_id = selected_item

                assert isinstance(group_id, int)
                assert isinstance(slice_id, int)

        # ----------------------------------------------------
        # 6.5 Check uniqueness inside each split
        # ----------------------------------------------------

        for split_name in (
            "training",
            "validation",
            "test",
        ):

            selected_slices = split_selection[split_name]

            assert len(selected_slices) == len(
                set(selected_slices)
            ), (
                f"Duplicate (group_id, slice_id) found: "
                f"domain={domain}, "
                f"file={file_name}, "
                f"split={split_name}"
            )

        # ----------------------------------------------------
        # 6.6 Check selected slices belong to allocated groups
        # ----------------------------------------------------

        allocated_groups = group_allocation[domain][file_name]

        for split_name in (
            "training",
            "validation",
            "test",
        ):

            selected_slices = split_selection[split_name]

            allowed_group_ids = set(
                allocated_groups[split_name]
            )

            for group_id, slice_id in selected_slices:

                assert group_id in allowed_group_ids, (
                    f"Selected slice comes from a group "
                    f"not allocated to {split_name}: "
                    f"domain={domain}, "
                    f"file={file_name}, "
                    f"group_id={group_id}, "
                    f"slice_id={slice_id}"
                )

        # ----------------------------------------------------
        # 6.7 Check selected slice actually exists
        # ----------------------------------------------------

        group_dict = data_index[domain][file_name]

        for split_name in (
            "training",
            "validation",
            "test",
        ):

            for group_id, slice_id in split_selection[split_name]:

                assert group_id in group_dict, (
                    f"Selected group does not exist: "
                    f"domain={domain}, "
                    f"file={file_name}, "
                    f"group_id={group_id}"
                )

                assert slice_id in group_dict[group_id], (
                    f"Selected slice does not exist: "
                    f"domain={domain}, "
                    f"file={file_name}, "
                    f"group_id={group_id}, "
                    f"slice_id={slice_id}"
                )

        # ----------------------------------------------------
        # 6.8 Check split-level group isolation
        # ----------------------------------------------------

        training_groups = {
            group_id
            for group_id, _ in split_selection["training"]
        }

        validation_groups = {
            group_id
            for group_id, _ in split_selection["validation"]
        }

        test_groups = {
            group_id
            for group_id, _ in split_selection["test"]
        }

        assert training_groups.isdisjoint(validation_groups), (
            f"Training / validation group overlap: "
            f"domain={domain}, file={file_name}"
        )

        assert training_groups.isdisjoint(test_groups), (
            f"Training / test group overlap: "
            f"domain={domain}, file={file_name}"
        )

        assert validation_groups.isdisjoint(test_groups), (
            f"Validation / test group overlap: "
            f"domain={domain}, file={file_name}"
        )


print("Step 6: _select_slices() test passed.")


# ============================================================
# Step 7: Test _build_sample_data()
# ============================================================

sample_data = sample_processor._build_sample_data(
    filtered_data=filtered_data,
    slice_selection=slice_selection,
)

# ------------------------------------------------------------
# 7.1 Check top-level structure
# ------------------------------------------------------------

assert set(sample_data.keys()) == {
    "training",
    "validation",
    "test",
}

# ------------------------------------------------------------
# 7.2 Check exact final sample numbers
# ------------------------------------------------------------

assert len(sample_data["training"]) == sample_processor.train_size

assert len(sample_data["validation"]) == sample_processor.validation_size

assert len(sample_data["test"]) == sample_processor.test_size

# ------------------------------------------------------------
# 7.3 Build expected selected identities
#
# Identity:
# (domain, file_name, group_id, slice_id)
# ------------------------------------------------------------

expected_selected_lookup = {
    "training": set(),
    "validation": set(),
    "test": set(),
}

for domain, file_dict in slice_selection.items():

    for file_name, split_selection in file_dict.items():

        for split_name in (
            "training",
            "validation",
            "test",
        ):

            for group_id, slice_id in (
                split_selection[split_name]
            ):

                expected_selected_lookup[split_name].add(
                    (
                        domain,
                        file_name,
                        group_id,
                        slice_id,
                    )
                )

# ------------------------------------------------------------
# 7.4 Check final DataProcessContainer objects
# ------------------------------------------------------------

for split_name in (
    "training",
    "validation",
    "test",
):

    for data_container in sample_data[split_name]:

        assert isinstance(
            data_container,
            DataProcessContainer,
        )

        assert data_container.sample_info is not None

        assert "sample_id" in data_container.sample_info

# ------------------------------------------------------------
# 7.5 Check sample_id sequence
# ------------------------------------------------------------

for split_name in (
    "training",
    "validation",
    "test",
):

    sample_ids = [
        data_container.sample_info["sample_id"]
        for data_container in sample_data[split_name]
    ]

    assert sample_ids == list(
        range(len(sample_data[split_name]))
    ), (
        f"Invalid sample_id sequence for {split_name}"
    )

# ------------------------------------------------------------
# 7.6 Check final samples correspond exactly to selection
# ------------------------------------------------------------

actual_selected_lookup = {
    "training": set(),
    "validation": set(),
    "test": set(),
}

for split_name in (
    "training",
    "validation",
    "test",
):

    for data_container in sample_data[split_name]:

        metadata_info = data_container.metadata[
            "metadata_info"
        ]

        domain = metadata_info["domain"]

        file_name = metadata_info["file_name"]

        group_id = data_container.group_info[
            "group_id"
        ]

        slice_id = data_container.slice_info[
            "slice_id"
        ]

        selection_key = (
            domain,
            file_name,
            group_id,
            slice_id,
        )

        actual_selected_lookup[split_name].add(
            selection_key
        )

# ------------------------------------------------------------
# 7.7 Exact equality between selected and final samples
# ------------------------------------------------------------

for split_name in (
    "training",
    "validation",
    "test",
):

    assert (
        actual_selected_lookup[split_name]
        == expected_selected_lookup[split_name]
    ), (
        f"Final sample selection mismatch: "
        f"{split_name}"
    )

# ------------------------------------------------------------
# 7.8 Check no duplicate samples inside each split
# ------------------------------------------------------------

for split_name in (
    "training",
    "validation",
    "test",
):

    selected_keys = actual_selected_lookup[split_name]

    assert len(selected_keys) == len(
        sample_data[split_name]
    ), (
        f"Duplicate samples found in {split_name}"
    )

# ------------------------------------------------------------
# 7.9 Check train / validation / test isolation
# ------------------------------------------------------------

training_final_keys = actual_selected_lookup["training"]

validation_final_keys = actual_selected_lookup["validation"]

test_final_keys = actual_selected_lookup["test"]

assert training_final_keys.isdisjoint(
    validation_final_keys
), "Training / validation sample overlap detected."

assert training_final_keys.isdisjoint(
    test_final_keys
), "Training / test sample overlap detected."

assert validation_final_keys.isdisjoint(
    test_final_keys
), "Validation / test sample overlap detected."

# ------------------------------------------------------------
# 7.10 Check sample_info only exists for selected samples
# ------------------------------------------------------------

selected_container_ids = set()

for split_name in (
    "training",
    "validation",
    "test",
):

    for data_container in sample_data[split_name]:

        selected_container_ids.add(
            id(data_container)
        )

for data_container in filtered_data:

    if id(data_container) in selected_container_ids:

        assert data_container.sample_info is not None

    else:

        assert data_container.sample_info is None, (
            "Unselected DataProcessContainer has "
            "unexpected sample_info."
        )

# ------------------------------------------------------------
# 7.11 Check final ordering follows filtered_data order
# ------------------------------------------------------------

filtered_data_order = {}

for index, data_container in enumerate(filtered_data):

    metadata_info = data_container.metadata[
        "metadata_info"
    ]

    domain = metadata_info["domain"]

    file_name = metadata_info["file_name"]

    group_id = data_container.group_info[
        "group_id"
    ]

    slice_id = data_container.slice_info[
        "slice_id"
    ]

    key = (
        domain,
        file_name,
        group_id,
        slice_id,
    )

    filtered_data_order[key] = index

for split_name in (
    "training",
    "validation",
    "test",
):

    final_order = []

    for data_container in sample_data[split_name]:

        metadata_info = data_container.metadata[
            "metadata_info"
        ]

        domain = metadata_info["domain"]

        file_name = metadata_info["file_name"]

        group_id = data_container.group_info[
            "group_id"
        ]

        slice_id = data_container.slice_info[
            "slice_id"
        ]

        key = (
            domain,
            file_name,
            group_id,
            slice_id,
        )

        final_order.append(
            filtered_data_order[key]
        )

    assert final_order == sorted(final_order), (
        f"Final {split_name} order does not follow "
        f"filtered_data order."
    )


print("Step 7: _build_sample_data() test passed.")


# ============================================================
# Final result
# ============================================================

print()
print("=" * 70)
print("SampleProcessor full sampling test passed.")
print("=" * 70)

print(
    f"Training samples:   {len(sample_data['training'])}"
)

print(
    f"Validation samples: {len(sample_data['validation'])}"
)

print(
    f"Test samples:       {len(sample_data['test'])}"
)