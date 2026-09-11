from copy import deepcopy

import numpy as np

from DataProcessPart.BenchmarkData import BenchmarkData


class Group_Processor:
    """Split raw signals into fixed-length, non-overlapping groups."""

    def __init__(self, group_size: int):
        """Initialize the group processor."""

        if group_size <= 0:
            raise ValueError(
                "group_size must be greater than 0."
            )

        self.group_size = group_size

    def process(
        self,
        data_list: list[BenchmarkData]
    ) -> list[BenchmarkData]:
        """Split all BenchmarkData objects into groups."""

        grouped_data = []

        for data in data_list:
            groups = self._process_one(data)
            grouped_data.extend(groups)

        return grouped_data

    def _process_one(
        self,
        data: BenchmarkData
    ) -> list[BenchmarkData]:
        """Split one BenchmarkData object into groups."""

        signal = data.X
        signal_length = len(signal)

        # Calculate the number of complete groups.
        num_groups = signal_length // self.group_size

        groups = []

        for group_id in range(num_groups):

            # Calculate the absolute start and end positions.
            start = group_id * self.group_size
            end = start + self.group_size - 1

            # Extract one complete group.
            group_signal = signal[start:end + 1]

            # Inherit all metadata from the source data.
            metadata = deepcopy(data.metadata)

            # Update the processing status.
            metadata["status"] = "grouped"

            # Record group-specific information.
            metadata["group_info"] = {
                "group_id": group_id,
                "str_end": f"[{start}-{end}]"
            }

            groups.append(
                BenchmarkData(
                    X=np.asarray(group_signal),
                    y=None,
                    metadata=metadata
                )
            )

        return groups