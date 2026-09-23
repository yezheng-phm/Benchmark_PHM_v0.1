#-----------------------------------------------------------------------
#--used to organize slices into samples according to the specified
#--training, validation, and test protocols.
#-----------------------------------------------------------------------

from pathlib import Path

import numpy as np
import torch

from DataProcessPart.DataProcessContainer import DataProcessContainer


#---------------------------sample processor logic configurations--------------------------------------

TRAIN_DOMAIN = [1797, 1797, 1750]
VALIDATION_DOMAIN = [1730]
TEST_DOMAIN = [1730]

TRAIN_SIZE = 2000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

GROUP_SLICE_SHUFFLE_SEED = 42

EXCLUDED_FAULT_SIZES = ["028"]

OR_FAULT_POSITIONS = [6]

SAMPLE_DATA_SAVE_PATH = r"D:\Project\PHM_Data\SampledData"
SAMPLE_DATA_SAVE_NAME = "CWRU_test.pt"

#------------------------------------------------------------------------------------------------------


class Sample_Processor:
    """Organize slices into training, validation, and test samples."""

    def __init__(
        self,
        data: list[DataProcessContainer],
        train_domain=TRAIN_DOMAIN,
        validation_domain=VALIDATION_DOMAIN,
        test_domain=TEST_DOMAIN,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        group_slice_shuffle_seed=GROUP_SLICE_SHUFFLE_SEED,
        excluded_fault_sizes=EXCLUDED_FAULT_SIZES,
        or_fault_positions=OR_FAULT_POSITIONS,
        sample_data_save_path=SAMPLE_DATA_SAVE_PATH,
        sample_data_save_name=SAMPLE_DATA_SAVE_NAME
    ):
        self.data = data
        self.train_domain = train_domain
        self.validation_domain = validation_domain
        self.test_domain = test_domain

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

        self.group_slice_shuffle_seed = group_slice_shuffle_seed
        self.excluded_fault_sizes = excluded_fault_sizes
        self.or_fault_positions = or_fault_positions
        self.sample_data_save_path = sample_data_save_path
        self.sample_data_save_name = sample_data_save_name

        self.rng = np.random.default_rng(
            self.group_slice_shuffle_seed
        )

#----------------------------------------------------------------------------------------------------

    #--filter input sliced data list to the sample processing protocol.
    def _filter_data(
        self,
    ) -> list[DataProcessContainer]:
        """Filter input slices according to the sample processing protocol."""

        filtered_data = []

        for data_container in self.data:

            metadata_info = data_container.metadata["metadata_info"]

            fault_size = metadata_info["fault_size"]
            fault_label = metadata_info["fault_label"]
            fault_position_num = metadata_info["fault_position_num"]

            if fault_size in self.excluded_fault_sizes:
                continue

            if (
                fault_label == "OR"
                and fault_position_num not in self.or_fault_positions
            ):
                continue

            filtered_data.append(data_container)

        return filtered_data



    #--build the index as "Domain -> File_name -> Slice" for sampling. 
    def _build_data_index(
        self,
        filtered_data: list[DataProcessContainer],
    ) -> dict:
        """Build the Domain -> File -> Group -> Slice index."""

        data_index = {}

        for data_container in filtered_data:

            domain = data_container.metadata["metadata_info"]["domain"]
            file_name = data_container.metadata["metadata_info"]["file_name"]
            group_id = data_container.group_info["group_id"]
            slice_id = data_container.slice_info["slice_id"]

            if domain not in data_index:
                data_index[domain] = {}

            if file_name not in data_index[domain]:
                data_index[domain][file_name] = {}

            if group_id not in data_index[domain][file_name]:
                data_index[domain][file_name][group_id] = []

            data_index[domain][file_name][group_id].append(
                slice_id
            )

        return data_index



    #--Distribute a total quota as evenly as possible.
    #--for example,1001/3= 333 334 334, is not 333 333 335.
    def _distribute_quota(
        self,
        total_quota: int,
        num_parts: int,
    ) -> list[int]:
        """Distribute a total quota as evenly as possible."""

        if total_quota < 0:
            raise ValueError(
                "total_quota must be non-negative."
            )

        if num_parts <= 0:
            raise ValueError(
                "num_parts must be greater than zero."
            )

        base_quota = total_quota // num_parts
        remainder = total_quota % num_parts

        quotas = [
            base_quota + 1
            if index < remainder
            else base_quota
            for index in range(num_parts)
        ]

        return quotas



    #--calculate sample requirments for each file.
    def _calculate_file_sample_requirements(
        self,
        data_index: dict,
    ) -> dict:
        """Calculate sample requirements for each file."""

        file_sample_requirements = {}

        split_configs = {
            "training": (
                self.train_domain,
                self.train_size,
            ),
            "validation": (
                self.validation_domain,
                self.validation_size,
            ),
            "test": (
                self.test_domain,
                self.test_size,
            ),
        }

        for split_name, (domains, total_size) in split_configs.items():

            if not domains:
                continue

            # Count how many times each domain appears.
            domain_counts = {}

            for domain in domains:
                domain_counts[domain] = (
                    domain_counts.get(domain, 0) + 1
                )

            # Distribute the total sample size among
            # all domain occurrences as evenly as possible.
            domain_occurrence_quotas = self._distribute_quota(
                total_quota=total_size,
                num_parts=len(domains),
            )

            # Assign each occurrence quota to its domain.
            domain_sample_requirements = {}

            occurrence_index = 0

            for domain in domains:

                domain_quota = domain_occurrence_quotas[
                    occurrence_index
                ]

                if domain not in domain_sample_requirements:
                    domain_sample_requirements[domain] = 0

                domain_sample_requirements[domain] += domain_quota

                occurrence_index += 1

            # Distribute each domain's quota among its files.
            for domain, domain_quota in domain_sample_requirements.items():

                if domain not in data_index:
                    raise ValueError(
                        f"Domain '{domain}' required for "
                        f"{split_name} but not found in data index."
                    )

                file_names = list(data_index[domain].keys())

                if not file_names:
                    raise ValueError(
                        f"No files found for domain '{domain}' "
                        f"required for {split_name}."
                    )

                file_quotas = self._distribute_quota(
                    total_quota=domain_quota,
                    num_parts=len(file_names),
                )

                for file_name, file_quota in zip(
                    file_names,
                    file_quotas,
                ):

                    if domain not in file_sample_requirements:
                        file_sample_requirements[domain] = {}

                    if file_name not in file_sample_requirements[domain]:
                        file_sample_requirements[domain][file_name] = {
                            "training": 0,
                            "validation": 0,
                            "test": 0,
                        }

                    file_sample_requirements[domain][file_name][
                        split_name
                    ] = file_quota

        return file_sample_requirements



    #--allocate complete groups to training, validation, and test
    #--within each file, groups are shuffled once and cannot cross splits.
    def _allocate_groups(
        self,
        data_index: dict,
        file_sample_requirements: dict,
    ) -> dict:
        """Allocate complete groups to training, validation, and test."""

        group_allocation = {}

        for domain, file_requirements in file_sample_requirements.items():

            group_allocation[domain] = {}

            for file_name, split_requirements in file_requirements.items():

                group_dict = data_index[domain][file_name]

                group_ids = list(group_dict.keys())

                if not group_ids:
                    raise ValueError(
                        f"No groups found for domain '{domain}', "
                        f"file '{file_name}'."
                    )

                # Shuffle groups once for this file.
                shuffled_group_ids = group_ids.copy()
                self.rng.shuffle(shuffled_group_ids)

                group_allocation[domain][file_name] = {
                    "training": [],
                    "validation": [],
                    "test": [],
                }

                remaining_group_ids = shuffled_group_ids.copy()

                split_configs = [
                    ("training", split_requirements["training"]),
                    ("validation", split_requirements["validation"]),
                    ("test", split_requirements["test"]),
                ]

                for split_name, required_samples in split_configs:

                    if required_samples == 0:
                        continue

                    current_capacity = 0

                    while (
                        current_capacity < required_samples
                        and remaining_group_ids
                    ):
                        group_id = remaining_group_ids.pop(0)

                        group_allocation[domain][file_name][
                            split_name
                        ].append(group_id)

                        current_capacity += len(
                            group_dict[group_id]
                        )

                    if current_capacity < required_samples:
                        raise ValueError(
                            f"Insufficient group capacity for "
                            f"{split_name}: "
                            f"domain='{domain}', "
                            f"file='{file_name}', "
                            f"required={required_samples}, "
                            f"capacity={current_capacity}."
                        )

        return group_allocation



    #--select the exact number of slices required by each split.
    #--each selected slice is stored together with its group_id.
    def _select_slices(
        self,
        data_index: dict,
        file_sample_requirements: dict,
        group_allocation: dict,
    ) -> dict:
        """Select the exact number of slices for each split."""

        slice_selection = {}

        for domain, file_requirements in file_sample_requirements.items():

            slice_selection[domain] = {}

            for file_name, split_requirements in file_requirements.items():

                group_dict = data_index[domain][file_name]
                allocated_groups = group_allocation[domain][file_name]

                slice_selection[domain][file_name] = {
                    "training": [],
                    "validation": [],
                    "test": [],
                }

                split_configs = [
                    ("training", split_requirements["training"]),
                    ("validation", split_requirements["validation"]),
                    ("test", split_requirements["test"]),
                ]

                for split_name, required_samples in split_configs:

                    if required_samples == 0:
                        continue

                    #--------------------------------------------------
                    #--Build candidate (group_id, slice_id) pairs.
                    #--------------------------------------------------

                    candidate_slices = []

                    for group_id in allocated_groups[split_name]:

                        for slice_id in group_dict[group_id]:

                            candidate_slices.append(
                                (group_id, slice_id)
                            )

                    #--------------------------------------------------
                    #--Validate candidate capacity.
                    #--------------------------------------------------

                    if len(candidate_slices) < required_samples:
                        raise ValueError(
                            f"Insufficient slice capacity for "
                            f"{split_name}: "
                            f"domain='{domain}', "
                            f"file='{file_name}', "
                            f"required={required_samples}, "
                            f"available={len(candidate_slices)}."
                        )

                    #--------------------------------------------------
                    #--Randomly select the required number of
                    #--(group_id, slice_id) pairs.
                    #--------------------------------------------------

                    selected_indices = self.rng.choice(
                        len(candidate_slices),
                        size=required_samples,
                        replace=False,
                    )

                    selected_slices = [
                        candidate_slices[index]
                        for index in selected_indices
                    ]

                    slice_selection[domain][file_name][
                        split_name
                    ] = selected_slices

        return slice_selection



    #--build the final training, validation, and test data lists.
    #--data are appended according to the order of filtered_data.
    #--sample_id is assigned sequentially within each split.
    def _build_sample_data(
        self,
        filtered_data: list[DataProcessContainer],
        slice_selection: dict,
    ) -> dict:
        """Build final training, validation, and test sample lists."""

        selected_slice_lookup = {
            "training": set(),
            "validation": set(),
            "test": set(),
        }

        #--------------------------------------------------------------
        #--Build lookup:
        #--(domain, file_name, group_id, slice_id)
        #--------------------------------------------------------------

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

                        selected_slice_lookup[
                            split_name
                        ].add(
                            (
                                domain,
                                file_name,
                                group_id,
                                slice_id,
                            )
                        )

        sample_data = {
            "training": [],
            "validation": [],
            "test": [],
        }

        sample_ids = {
            "training": 0,
            "validation": 0,
            "test": 0,
        }

        #--------------------------------------------------------------
        #--Find the original DataProcessContainer from filtered_data.
        #--The order of filtered_data determines the final order.
        #--------------------------------------------------------------

        for data_container in filtered_data:

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

            for split_name in (
                "training",
                "validation",
                "test",
            ):

                if (
                    selection_key
                    not in selected_slice_lookup[split_name]
                ):
                    continue

                #------------------------------------------------------
                #--This Slice is officially selected as a Sample.
                #------------------------------------------------------

                data_container.sample_info = {
                    "sample_id": sample_ids[split_name]
                }

                sample_data[split_name].append(
                    data_container
                )

                sample_ids[split_name] += 1

                break

        return sample_data



    def _save_sample_data(self, sample_data: dict) -> None:
        """Save the sampled training, validation, and test data."""

        save_path = Path(self.sample_data_save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / self.sample_data_save_name

        torch.save(sample_data, file_path)


    def run(self) -> dict:
        """Run the complete sample processing pipeline and save the result."""

        filtered_data = self._filter_data()

        data_index = self._build_data_index(
            filtered_data
        )

        sample_requirements = self._calculate_file_sample_requirements(
            data_index
        )

        allocated_groups = self._allocate_groups(
            data_index,
            sample_requirements
        )

        selected_slices = self._select_slices(
            data_index,
            sample_requirements,
            allocated_groups
        )

        sample_data = self._build_sample_data(
            filtered_data,
            selected_slices
        )

        self._save_sample_data(
            sample_data
        )

        return sample_data