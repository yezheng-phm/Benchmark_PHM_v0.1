#----------------------------------------------------------------------
#--this file is used to group current data with Group_logic.
#----------------------------------------------------------------------

from copy import deepcopy

import numpy as np


from DataProcessPart.DataProcessContainer import DataProcessContainer



#-------------------set group logic configs-----------
MIN_GROUP_SIZE = 6000
MAX_GROUP_SIZE = 8000
#-----------------------------------------------------



class Group_Processor:
    """Split signals into non-overlapping groups with dynamically selected sizes."""

    def __init__(
            self,
            data_list: list[DataProcessContainer],
    ):
        self._validate_group_size_range(
            MIN_GROUP_SIZE,
            MAX_GROUP_SIZE,
        )

        self.min_group_size = MIN_GROUP_SIZE
        self.max_group_size = MAX_GROUP_SIZE

        self.data_list = data_list


    #--validate min_group_size and max_group_size are legal.
    def _validate_group_size_range(
        self,
        min_group_size: int,
        max_group_size: int,
    ) -> None:
        """Validate the configured group size range."""

        if min_group_size <= 0:
            raise ValueError(
                "min_group_size must be greater than 0."
            )

        if max_group_size <= 0:
            raise ValueError(
                "max_group_size must be greater than 0."
            )

        if min_group_size > max_group_size:
            raise ValueError(
                "min_group_size must not be greater than "
                "max_group_size. "
                f"Current values are "
                f"min_group_size={min_group_size}, "
                f"max_group_size={max_group_size}."
            )


    #--validate the current data length is enough for grouping.
    def _validate_data_length(
        self,
        data: DataProcessContainer,
    ) -> None:
        """Validate that the current data is long enough for grouping."""

        data_length = len(data.X_data)

        if data_length < self.min_group_size:
            raise ValueError(
                "Current data length is smaller than min_group_size. "
                f"Current data length={data_length}, "
                f"min_group_size={self.min_group_size}."
            )


    #--find the optimal group size for current data.
    def _find_optimal_group_size(
        self,
        data: DataProcessContainer,
    ) -> int:
        """Find the optimal group size for the current data."""

        data_length = len(data.X_data)

        max_group_size = min(
            self.max_group_size,
            data_length,
        )

        optimal_group_size = self.min_group_size
        min_remainder = data_length % optimal_group_size

        for group_size in range(
            self.min_group_size,
            max_group_size + 1,
        ):
            remainder = data_length % group_size

            if remainder <= min_remainder:
                min_remainder = remainder
                optimal_group_size = group_size

        return optimal_group_size


    #--core method logic!!
    #--create non-overlapping groups from current data.
    def _create_groups(
        self,
        data: DataProcessContainer,
        optimal_group_size: int,
    ) -> list[DataProcessContainer]:
        """Create non-overlapping groups from the current data."""

        data_length = len(data.X_data)

        num_groups = data_length // optimal_group_size

        groups = []

        for group_id in range(num_groups):

            start = group_id * optimal_group_size
            end = start + optimal_group_size - 1

            group_data = data.X_data[start:end + 1]

            group_info = {
                "group_id": group_id,
                "start": start,
                "end": end,
                "group_size": optimal_group_size,
            }

            group_container = DataProcessContainer(
                X_data=np.asarray(group_data),
                y=data.y,
                metadata=deepcopy(data.metadata),
                group_info=group_info,
                slice_info=None,
                sample_info=None,
                extra_info=deepcopy(data.extra_info),
            )

            groups.append(group_container)

        return groups


    #--process one DataProcessContainer.
    def _single_process(
        self,
        data: DataProcessContainer,
    ) -> list[DataProcessContainer]:
        """Process one data container into groups."""

        self._validate_data_length(data)

        optimal_group_size = self._find_optimal_group_size(data)

        groups = self._create_groups(
            data,
            optimal_group_size,
        )

        self._print_group_summary(
            data,
            optimal_group_size,
            groups,
        )

        return groups


    #--process all DataProcessContainer objects.
    def _total_group_process(
        self,
    ) -> list[DataProcessContainer]:
        """Process all data containers into non-overlapping groups."""

        grouped_data = []

        for data in self.data_list:

            groups = self._single_process(data)

            grouped_data.extend(groups)

        return grouped_data



    #--print the core group processing information.
    def _print_group_summary(
        self,
        data: DataProcessContainer,
        optimal_group_size: int,
        groups: list[DataProcessContainer],
    ) -> None:
        """Print the core information of the group processing result."""

        data_length = len(data.X_data)

        group_count = len(groups)

        discarded_points = (
            data_length
            - group_count * optimal_group_size
        )

        file_name = data.metadata["metadata_info"]["file_name"]

        print(
            f"[Group] "
            f"File={file_name} | "
            f"Length={data_length} | "
            f"Group Size={optimal_group_size} | "
            f"Groups={group_count} | "
            f"Discarded={discarded_points}"
        )


    #--run the complete group processing pipeline.
    def run(
        self,
    ) -> list[DataProcessContainer]:
        """Run the complete group processing pipeline."""

        grouped_data = self._total_group_process()

        return grouped_data




    






















        