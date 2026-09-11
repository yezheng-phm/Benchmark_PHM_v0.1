from copy import deepcopy

import numpy as np

from DataProcessPart.BenchmarkData import BenchmarkData


class Slice_Processor:
    """Split groups into fixed-length slices."""

    def __init__(
        self,
        slice_size: int,
        slice_stride: int
    ):
        """Initialize the slice processor."""

        # Both slice parameters must be positive.
        if slice_size <= 0:
            raise ValueError(
                "slice_size must be greater than 0."
            )

        if slice_stride <= 0:
            raise ValueError(
                "slice_stride must be greater than 0."
            )

        self.slice_size = slice_size
        self.slice_stride = slice_stride

    def _calculate_overlap_rate(self) -> float:
        """Calculate the overlap rate between adjacent slices."""

        if self.slice_stride >= self.slice_size:
            return 0.0

        return (
            1
            - self.slice_stride / self.slice_size
        )

    def _process_one(
        self,
        data: BenchmarkData
    ) -> list[BenchmarkData]:
        """Split one Group BenchmarkData into slices."""

        signal = data.X
        group_length = len(signal)

        # slice_size cannot be larger than the Group.
        if self.slice_size > group_length:
            raise ValueError(
                f"slice_size ({self.slice_size}) "
                f"cannot be greater than group length "
                f"({group_length})."
            )

        # Get the absolute start position of the Group.
        group_info = data.metadata["group_info"]
        group_start = int(
            group_info["str_end"]
            .strip("[]")
            .split("-")[0]
        )

        # Calculate the overlap rate once for this processor.
        overlap_rate = self._calculate_overlap_rate()

        slices = []

        slice_id = 0
        start = 0

        while start + self.slice_size <= group_length:

            # Calculate the end position inside the Group.
            end = start + self.slice_size - 1

            # Extract the slice signal.
            slice_signal = signal[start:end + 1]

            # Copy all metadata from the Group.
            metadata = deepcopy(data.metadata)

            # Update the processing status.
            metadata["status"] = "sliced"

            # Convert the local position to the original
            # signal's absolute position.
            absolute_start = group_start + start
            absolute_end = group_start + end

            # Create new slice information.
            metadata["slice_info"] = {
                "slice_id": slice_id,
                "str_end": (
                    f"[{absolute_start}-{absolute_end}]"
                ),
                "slice_size": self.slice_size,
                "slice_stride": self.slice_stride,
                "overlap_rate": overlap_rate
            }

            slices.append(
                BenchmarkData(
                    X=np.asarray(slice_signal),
                    y=None,
                    metadata=metadata
                )
            )

            slice_id += 1
            start += self.slice_stride

        return slices

    def process(
        self,
        data_list: list[BenchmarkData]
    ) -> list[BenchmarkData]:
        """Split all Group BenchmarkData objects into slices."""

        sliced_data = []

        # Process Groups in their existing order.
        for data in data_list:

            slices = self._process_one(data)

            sliced_data.extend(slices)

        return sliced_data