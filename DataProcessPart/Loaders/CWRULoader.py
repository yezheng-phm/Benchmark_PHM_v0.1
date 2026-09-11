from pathlib import Path

import numpy as np
from scipy.io import loadmat

from DataProcessPart.BenchmarkData import BenchmarkData


class CWRULoader:
    """Load raw data from the CWRU dataset."""

    def __init__(self, root_dir: str):
        """Initialize the loader with the dataset root directory."""

        self.root_dir = Path(root_dir)

    def load(self) -> list[BenchmarkData]:
        """Load all .mat files from the dataset."""

        data_list = []

        mat_files = self._scan_files()

        for file_path in mat_files:

            # Separate normal and fault data.
            if "Normal" in file_path.parts:
                data = self._load_normal_file(file_path)
            else:
                data = self._load_fault_file(file_path)

            data_list.append(data)

        return data_list

    def _scan_files(self) -> list[Path]:
        """Find all .mat files under the root directory."""

        return list(self.root_dir.rglob("*.mat"))

    def _load_fault_file(
        self,
        file_path: Path
    ) -> BenchmarkData:
        """Load one fault data file."""

        mat_data = loadmat(file_path)

        # Find the DE signal used in the current fault file.
        signal = None
        key = None

        for current_key, value in mat_data.items():

            if current_key.endswith("_DE_time"):
                key = current_key
                signal = value.squeeze()
                break

        if signal is None:
            raise ValueError(
                f"No DE signal found in: {file_path}"
            )

        # Parse load and nominal RPM from the parent directory.
        condition = file_path.parent.name

        load_str, rpm_str = condition.split("_")

        load = int(load_str)
        rpm = int(rpm_str)

        # Parse and normalize fault size to three digits.
        fault_size = f"{int(float(file_path.parent.parent.name) * 1000):03d}"

        file_name = file_path.name

        # Determine the fault category from the file name.
        if file_name.startswith("IR"):
            fault_label = "IR"

        elif file_name.startswith("B"):
            fault_label = "B"

        elif file_name.startswith("OR"):
            fault_label = "OR"

        else:
            raise ValueError(
                f"Unknown fault type: {file_path}"
            )

        # Default values for fault position.
        fault_position = None
        fault_position_num = None

        # Parse the fault position for Outer Race faults.
        if fault_label == "OR":

            if "@3" in file_name:
                fault_position = "Orthogonal"
                fault_position_num = 3

            elif "@6" in file_name:
                fault_position = "Centered"
                fault_position_num = 6

            elif "@12" in file_name:
                fault_position = "Opposite"
                fault_position_num = 12

            else:
                raise ValueError(
                    f"Unknown fault position: {file_path}"
                )

        # Build unified metadata.
        metadata = {
            "status": "rawdata",

            "dataset": "CWRU",

            "info": {
                "fault_label": fault_label,
                "fault_size": fault_size,
                "rpm": rpm,
                "load": load,
                "channel": "DE",
                "file_name": file_name
            },

            "extra_info": {
                "fault_position": fault_position,
                "fault_position_num": fault_position_num,

                # Record the actual MATLAB variable used.
                "source_key": key
            }
        }

        return BenchmarkData(
            X=np.asarray(signal),
            y=None,
            metadata=metadata
        )

    def _load_normal_file(
        self,
        file_path: Path
    ) -> BenchmarkData:
        """Load one normal CWRU .mat file."""

        # Load all variables from the .mat file.
        mat_data = loadmat(file_path)

        # Manually select a DE signal when multiple DE signals exist.
        select_key = None

        if file_path.name == "Normal_2.mat":
            select_key = "X099_DE_time"

        # Find all Drive End signal keys.
        de_keys = [
            key
            for key in mat_data
            if key.endswith("_DE_time")
        ]

        if not de_keys:
            raise ValueError(
                f"No DE signal found in: {file_path}"
            )

        # Use the manually selected key if specified.
        if select_key is not None:

            if select_key not in de_keys:
                raise ValueError(
                    f"Selected key {select_key} "
                    f"not found in: {file_path}"
                )

            key = select_key

        # Automatically use the signal when only one DE exists.
        elif len(de_keys) == 1:

            key = de_keys[0]

        # Do not automatically choose when multiple DE signals exist.
        else:
            raise ValueError(
                f"Multiple DE signals found in: {file_path}. "
                f"Please specify select_key."
            )

        # Extract the selected DE signal.
        signal = mat_data[key].squeeze()

        file_name = file_path.name

        # RPM is defined by the parent directory.
        rpm = int(file_path.parent.name)

        # Load is extracted from Normal_x.mat.
        load = int(
            file_path.stem.split("_")[-1]
        )

        # Build unified metadata.
        metadata = {
            "status": "rawdata",

            "dataset": "CWRU",

            "info": {
                "fault_label": "Normal",
                "fault_size": None,
                "rpm": rpm,
                "load": load,
                "channel": "DE",
                "file_name": file_name
            },

            "extra_info": {
                "fault_position": None,
                "fault_position_num": None,

                # Record the actual MATLAB variable used.
                "source_key": key
            }
        }

        return BenchmarkData(
            X=np.asarray(signal),
            y=None,
            metadata=metadata
        )