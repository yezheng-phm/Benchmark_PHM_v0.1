#-----------------------------------------------------------------------
#--used to load CWRU dataset from specified path.
#--adapts raw data into DataProcessContainer objects.
#-----------------------------------------------------------------------

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from DataProcessPart.DataProcessContainer import DataProcessContainer


class CWRU_Loader:
    """Load and adapt raw data from the CWRU dataset."""

    def __init__(
        self,
        raw_data_root_path: str,
        data_set_name: str,
        current_data_path: str,
    ):
        self.raw_data_root_path = Path(raw_data_root_path)
        self.data_set_name = data_set_name
        self.current_data_path = Path(current_data_path)

        self._validate_data_directory()


    #--validata data direcory is useful or not.
    def _validate_data_directory(self) -> None:
        """Validate that the specified data directory is not empty."""

        data_path = (
            self.raw_data_root_path / self.current_data_path
        )

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {data_path}"
            )

        if not data_path.is_dir():
            raise NotADirectoryError(
                f"Specified data path is not a directory: {data_path}"
            )

        if not any(data_path.iterdir()):
            raise ValueError(
                f"Data directory is empty: {data_path}"
            )


    #--read all CWRU data from the specified path.
    def _read_data_files(self) -> list:
        """Read all MAT files from the current data directory."""

        data_path = self.raw_data_root_path / self.current_data_path

        data_files = sorted(data_path.rglob("*.mat"))

        data_list = []

        for file_path in data_files:
            data = loadmat(file_path)

            relative_path = (
                Path(self.data_set_name)
                / file_path.relative_to(data_path)
            )

            data_list.append((relative_path, data))

        return data_list


    #--validate source key is legal or not.
    def _validate_vibration_source_key(
        self,
        relative_path: Path,
        source_key: str | None,
        mat_data: dict,
    ) -> None:
        """Validate the vibration source key."""

        if source_key is None:
            raise ValueError(
                f"Unable to determine vibration source key: {relative_path}"
            )

        if source_key not in mat_data:
            raise KeyError(
                f"Vibration source key '{source_key}' "
                f"not found in: {relative_path}"
            )

        vibration_data = mat_data[source_key]

        if vibration_data is None or np.asarray(vibration_data).size == 0:
            raise ValueError(
                f"Vibration data is empty for source key "
                f"'{source_key}': {relative_path}"
            )


    #--get vibration source key from vibration signal.
    def _get_vibration_source_key(
        self,
        relative_path: Path,
        mat_data: dict,
    ) -> str:
        """Get and validate the source key of the raw DE vibration signal."""

        file_name = relative_path.name

        # Normal data
        if file_name.startswith("Normal_"):
            normal_source_keys = {
                "Normal_0.mat": "X097_DE_time",
                "Normal_1.mat": "X098_DE_time",
                "Normal_2.mat": "X099_DE_time",
                "Normal_3.mat": "X100_DE_time",
            }

            source_key = normal_source_keys.get(file_name)

        # Fault data
        else:
            source_key = next(
                (
                    key
                    for key in mat_data
                    if key.endswith("_DE_time")
                ),
                None,
            )

        self._validate_vibration_source_key(
            relative_path,
            source_key,
            mat_data,
        )

        return source_key


    #--get the raw vibration data from current .mat data file.
    def _get_raw_vibration_data(
        self,
        relative_path: Path,
        mat_data: dict,
    ) -> np.ndarray:
        """Extract the raw DE vibration signal."""

        source_key = self._get_vibration_source_key(
            relative_path,
            mat_data,
        )

        return np.asarray(mat_data[source_key]).squeeze()


    #--get semantic class label of the current sample.
    def _get_y(
        self,
        relative_path: Path,
    ) -> str:
        """Get the semantic class label of the current sample."""

        path_parts = relative_path.parts

        # Normal data
        if "Normal" in path_parts:
            return "Normal"

        # Fault data
        fault_size = path_parts[-3]
        file_name = path_parts[-1]

        fault_size = f"{int(float(fault_size) * 1000):03d}"

        fault_label = file_name.split(fault_size)[0]

        return f"{fault_size}-{fault_label}"


    #--get metadata for current vibration data.
    def _get_metadata(
        self,
        relative_path: Path,
        mat_data: dict,
    ) -> dict:

        path_parts = relative_path.parts

        source_key = self._get_vibration_source_key(
            relative_path,
            mat_data,
        )

        # Normal data
        if "Normal" in path_parts:
            rpm = int(path_parts[-2])
            load = int(relative_path.stem.split("_")[1])

            metadata_info = {
                "domain": rpm,
                "file_name": relative_path.name,
                "rpm": rpm,
                "load": load,
                "fault_size": None,
                "fault_label": "Normal",
                "source_key": source_key,
                "fault_position": None,
                "fault_position_num": None,
            }

        # Fault data
        else:
            fault_size = path_parts[-3]
            load_rpm = path_parts[-2]
            file_name = path_parts[-1]

            load, rpm = map(int, load_rpm.split("_"))

            fault_size = f"{int(float(fault_size) * 1000):03d}"

            fault_label = file_name.split(fault_size)[0]

            fault_position = None
            fault_position_num = None

            if fault_label == "OR":
                position_map = {
                    "@3": ("Orthogonal", 3),
                    "@6": ("Centered", 6),
                    "@12": ("Opposite", 12),
                }

                for position_key, position_value in position_map.items():
                    if position_key in file_name:
                        fault_position, fault_position_num = position_value
                        break

            metadata_info = {
                "domain": rpm,
                "file_name": relative_path.name,
                "rpm": rpm,
                "load": load,
                "fault_size": fault_size,
                "fault_label": fault_label,
                "source_key": source_key,
                "fault_position": fault_position,
                "fault_position_num": fault_position_num,
            }

        return {
            "data_set_name": self.data_set_name,
            "metadata_info": metadata_info,
        }


    #--create data container list from the loaded CWRU data.
    def _create_data_containers(
        self,
        data_list: list,
    ) -> list[DataProcessContainer]:
        """Create data containers from the loaded CWRU data."""

        data_containers = []

        for relative_path, mat_data in data_list:

            raw_vibration_data = self._get_raw_vibration_data(
                relative_path,
                mat_data,
            )

            y = self._get_y(
                relative_path
            )

            metadata = self._get_metadata(
                relative_path,
                mat_data,
            )

            data_container = DataProcessContainer(
                X_data=raw_vibration_data,
                y=y,
                metadata=metadata,
                extra_info=None,
            )

            data_containers.append(data_container)

        return data_containers


    #--run the CWRU data loading process.
    def run(self) -> list[DataProcessContainer]:
        """Run the CWRU data loading process."""

        data_list = self._read_data_files()

        data_containers = self._create_data_containers(
            data_list
        )

        return data_containers