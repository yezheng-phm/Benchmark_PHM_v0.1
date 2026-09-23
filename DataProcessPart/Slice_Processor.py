#--------------------------------------------------------------------
#--this file is used to slice grouped data into specified sizes.
#--the sliced data will be used by the Sample_Processor to
#--construct the final samples.
#--------------------------------------------------------------------

from copy import deepcopy

import numpy as np

from DataProcessPart.DataProcessContainer import DataProcessContainer

#-------------------set slice logic configs-----------
SLICE_SIZE = 1024
OVERLAP_SIZE = 512
MIN_REMAINING_SIZE = 10
#-----------------------------------------------------


class Slice_Processor:
    """Split grouped data into fixed-length slices."""

    def __init__(
        self,
        data_list: list[DataProcessContainer],
    ):

        self.data_list = data_list
        self.slice_size = SLICE_SIZE
        self.overlap_size = OVERLAP_SIZE

        self._validate_slice_parameters()




    #--validate slice parameters.
    def _validate_slice_parameters(
        self,
    ) -> None:
        """Validate the configured slice parameters."""

        if self.slice_size <= 0:
            raise ValueError(
                "slice_size must be greater than 0.\n"
                f"Current value: slice_size={self.slice_size}"
            )

        if self.overlap_size < 0:
            raise ValueError(
                "overlap_size must be greater than or equal to 0.\n"
                f"Current value: overlap_size={self.overlap_size}"
            )

        if self.overlap_size >= self.slice_size:
            raise ValueError(
                "overlap_size must be smaller than slice_size.\n"
                f"Current values: slice_size={self.slice_size}, "
                f"overlap_size={self.overlap_size}"
            )


    #--apply reflect padding to the current slice.
    def _apply_reflect_padding(
        self,
        signal: np.ndarray,
    ) -> np.ndarray | None:
        """Apply reflect padding to complete the current slice."""

        signal_length = len(signal)

        if signal_length == self.slice_size:
            return np.asarray(signal)

        if signal_length <= MIN_REMAINING_SIZE:
            return None

        padding_size = (
            self.slice_size
            - signal_length
        )

        return np.pad(
            signal,
            (0, padding_size),
            mode="reflect",
        )



    #--core method logic!!
    #--create fixed-length slices from current Group.
    def _create_slices(
        self,
        data: DataProcessContainer,
    ) -> list[DataProcessContainer]:
        signal = data.X_data
        group_length = len(signal)

        group_info = data.group_info
        group_start = group_info["start"]

        slice_stride = (
            self.slice_size
            - self.overlap_size
        )

        slices = []

        slice_id = 0
        start = 0

        # Create complete slices
        while start + self.slice_size <= group_length:

            end = start + self.slice_size

            slice_signal = signal[start:end]

            absolute_start = group_start + start
            absolute_end = (
                group_start
                + end
                - 1
            )

            slice_info = {
                "slice_id": slice_id,
                "start": absolute_start,
                "end": absolute_end,
                "slice_size": self.slice_size,
                "is_not_padded": True,
            }

            slice_container = DataProcessContainer(
                X_data=np.asarray(slice_signal),
                y=data.y,
                metadata=deepcopy(data.metadata),
                group_info=deepcopy(data.group_info),
                slice_info=slice_info,
                sample_info=None,
                extra_info=deepcopy(data.extra_info),
            )

            slices.append(slice_container)

            slice_id += 1
            start += slice_stride

        # Handle the remaining tail
        remaining_length = group_length - start

        if remaining_length > MIN_REMAINING_SIZE:

            slice_signal = signal[start:group_length]

            processed_signal = self._apply_reflect_padding(
                slice_signal
            )

            absolute_start = group_start + start
            absolute_end = (
                group_start
                + group_length
                - 1
            )

            slice_info = {
                "slice_id": slice_id,
                "start": absolute_start,
                "end": absolute_end,
                "slice_size": self.slice_size,
                "is_not_padded": False,
            }

            slice_container = DataProcessContainer(
                X_data=np.asarray(processed_signal),
                y=data.y,
                metadata=deepcopy(data.metadata),
                group_info=deepcopy(data.group_info),
                slice_info=slice_info,
                sample_info=None,
                extra_info=deepcopy(data.extra_info),
            )

            slices.append(slice_container)

        return slices



    #--process one DataProcessContainer.
    def _single_process(
        self,
        data: DataProcessContainer,
    ) -> list[DataProcessContainer]:
        """Process one Group DataProcessContainer  into slices."""

        slices = self._create_slices(
            data
        )

        self._print_slice_summary(
            data,
            slices,
        )

        return slices



    #--print the core slice processing information.
    def _print_slice_summary(
        self,
        data: DataProcessContainer,
        slices: list[DataProcessContainer],
    ) -> None:
        """Print the core information of the slice processing result."""

        file_name = data.metadata["metadata_info"]["file_name"]
        group_id = data.group_info["group_id"]
        group_length = len(data.X_data)

        slice_count = len(slices)

        last_slice_padded = 0

        if slices:
            last_slice = slices[-1]

            if not last_slice.slice_info["is_not_padded"]:
                actual_length = (
                    last_slice.slice_info["end"]
                    - last_slice.slice_info["start"]
                    + 1
                )

                last_slice_padded = (
                    self.slice_size
                    - actual_length
                )

        print(
            f"[Slice] "
            f"File={file_name} | "
            f"Group={group_id} | "
            f"Length={group_length} | "
            f"Slice Size={self.slice_size} | "
            f"Slices={slice_count} | "
            f"Last Slice Padded={last_slice_padded}"
        )




    #--process all DataProcessContainer objects.
    def _total_slice_process(
        self,
    ) -> list[DataProcessContainer]:
        """Process all Group DataProcessContainer  objects into slices."""

        sliced_data = []

        print(
            "\n"
            "-------------------------------------Slice info ---------------------------------------"
        )

        for data in self.data_list:

            slices = self._single_process(
                data
            )

            sliced_data.extend(slices)

        return sliced_data


    #--run the complete slice processing pipeline.
    def run(
        self,
    ) -> list[DataProcessContainer]:
        """Run the complete slice processing pipeline."""

        sliced_data = self._total_slice_process()

        return sliced_data