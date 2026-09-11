from copy import deepcopy
from math import ceil
from pathlib import Path

from datetime import datetime
import numpy as np
import torch

from DataProcessPart.BenchmarkData import BenchmarkData


class Sample_Processor:
    """Select Slices and build train, validation, and test sets."""

    def __init__(
        self,
        train_conditions: list[int],
        validation_conditions: list[int],
        test_conditions: list[int],
        train_size: int,
        validation_size: int,
        test_size: int,
        excluded_fault_sizes: list[str],
        or_fault_position: int,
        random_seed: int,
        output_dir: str,
        output_folder: str,
    ):
        self.train_conditions = train_conditions
        self.validation_conditions = validation_conditions
        self.test_conditions = test_conditions

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

        self.excluded_fault_sizes = excluded_fault_sizes

        self.or_fault_position = or_fault_position
        self.rng = np.random.default_rng(random_seed)

        # Output path.
        self.output_dir = ( 
            Path(output_dir) / output_folder 
            )
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_file = (
        self.output_dir
        / f"sampled_data_{timestamp}.pt"
        )

    # ==========================================================
    # Main
    # ==========================================================

    def process(
        self,
        data_list: list[BenchmarkData],
    ):
        """Build train, validation, and test sample sets."""

        train_requirements = self._condition_requirements(
            self.train_conditions,
            self.train_size,
        )

        validation_requirements = self._condition_requirements(
            self.validation_conditions,
            self.validation_size,
        )

        test_requirements = self._condition_requirements(
            self.test_conditions,
            self.test_size,
        )

        train_sources = self._source_requirements(
            data_list,
            train_requirements,
        )

        validation_sources = self._source_requirements(
            data_list,
            validation_requirements,
        )

        test_sources = self._source_requirements(
            data_list,
            test_requirements,
        )

        sources = (
            set(train_sources)
            | set(validation_sources)
            | set(test_sources)
        )

        train_set = []
        validation_set = []
        test_set = []

        for source in sorted(
            sources,
            key=self._source_sort_key,
        ):

            source_data = self._filter_source(
                data_list,
                source,
            )

            train_size = train_sources.get(
                source,
                0,
            )

            validation_size = validation_sources.get(
                source,
                0,
            )

            test_size = test_sources.get(
                source,
                0,
            )

            train_samples, validation_samples, test_samples = (
                self._process_source(
                    source_data,
                    source,
                    train_size,
                    validation_size,
                    test_size,
                )
            )

            train_set.extend(train_samples)
            validation_set.extend(validation_samples)
            test_set.extend(test_samples)

        # Final global mixing.
        self.rng.shuffle(train_set)
        self.rng.shuffle(validation_set)
        self.rng.shuffle(test_set)

        self._finalize(train_set)
        self._finalize(validation_set)
        self._finalize(test_set)

        self._check_leakage(
            train_set,
            validation_set,
            test_set,
        )

        # Save processed data.
        self._save(
            train_set,
            validation_set,
            test_set,
        )

        return train_set, validation_set, test_set

    # ==========================================================
    # 1. Condition allocation
    # ==========================================================

    def _condition_requirements(
        self,
        conditions: list[int],
        total_size: int,
    ) -> dict[int, int]:
        """
        Allocate samples to Conditions.

        Repeated Conditions occupy repeated shares.

        Example:
            [1797, 1797, 1750]
            6000
            ->
            1797: 4000
            1750: 2000
        """

        if not conditions:
            raise ValueError(
                "Conditions cannot be empty."
            )

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
    # 2. Source allocation
    # ==========================================================

    def _source_requirements(
        self,
        data_list: list[BenchmarkData],
        condition_requirements: dict[int, int],
    ) -> dict[tuple, int]:
        """Allocate Condition samples to Sources."""

        requirements = {}

        for condition, total_size in (
            condition_requirements.items()
        ):

            condition_data = self._filter_condition(
                data_list,
                condition,
            )

            sources = self._discover_sources(
                condition_data
            )

            if not sources:
                raise ValueError(
                    f"No valid Source found for "
                    f"condition={condition}."
                )

            allocations = self._allocate(
                total_size,
                sources,
            )

            for source, amount in zip(
                sources,
                allocations,
            ):
                requirements[source] = amount

        return requirements

    # ==========================================================
    # 3. Source-level Group partition
    # ==========================================================

    def _process_source(
        self,
        source_data: list[BenchmarkData],
        source: tuple,
        train_size: int,
        validation_size: int,
        test_size: int,
    ):
        """Partition Groups and select Slices for one Source."""

        groups = {}

        for data in source_data:

            group_id = (
                data.metadata["group_info"]["group_id"]
            )

            groups.setdefault(
                group_id,
                []
            ).append(data)

        group_ids = sorted(groups)

        if not group_ids:
            raise ValueError(
                f"No Groups found for Source:\n"
                f"{self._format_source(source)}"
            )

        # Every Group should contain the same number of Slices.
        slice_counts = {
            len(slices)
            for slices in groups.values()
        }

        if len(slice_counts) != 1:
            raise ValueError(
                f"Inconsistent Slice counts between Groups.\n"
                f"Source:\n"
                f"{self._format_source(source)}\n"
                f"Slice counts: {sorted(slice_counts)}"
            )

        slices_per_group = slice_counts.pop()

        group_length = self._group_length(
            source_data[0]
        )

        available_groups = len(group_ids)

        train_groups = ceil(
            train_size / slices_per_group
        )

        validation_groups = ceil(
            validation_size / slices_per_group
        )

        test_groups = ceil(
            test_size / slices_per_group
        )

        total_required_groups = (
            train_groups
            + validation_groups
            + test_groups
        )

        # ------------------------------------------------------
        # Hard availability check.
        # ------------------------------------------------------

        if total_required_groups > available_groups:

            total_vibration_points = (
                available_groups * group_length
            )

            raise ValueError(
                "\n"
                "==================================================\n"
                "Insufficient Groups for Source\n"
                "==================================================\n"
                f"Source:\n"
                f"  {self._format_source(source)}\n"
                "\n"
                f"Total vibration points:\n"
                f"  {total_vibration_points}\n"
                "\n"
                f"Group length:\n"
                f"  {group_length}\n"
                "\n"
                f"Available Groups:\n"
                f"  {available_groups}\n"
                "\n"
                f"Slices per Group:\n"
                f"  {slices_per_group}\n"
                "\n"
                f"Train:\n"
                f"  Required Slices: {train_size}\n"
                f"  Required Groups: {train_groups}\n"
                "\n"
                f"Validation:\n"
                f"  Required Slices: {validation_size}\n"
                f"  Required Groups: "
                f"{validation_groups}\n"
                "\n"
                f"Test:\n"
                f"  Required Slices: {test_size}\n"
                f"  Required Groups: {test_groups}\n"
                "\n"
                f"Total Required Groups:\n"
                f"  {total_required_groups}\n"
                "\n"
                f"Available Groups:\n"
                f"  {available_groups}\n"
                "=================================================="
            )

        # ------------------------------------------------------
        # Randomly shuffle Groups ONCE.
        # ------------------------------------------------------

        shuffled_groups = self.rng.permutation(
            group_ids
        ).tolist()

        train_ids = shuffled_groups[
            :train_groups
        ]

        validation_start = train_groups

        validation_end = (
            validation_start
            + validation_groups
        )

        validation_ids = shuffled_groups[
            validation_start:validation_end
        ]

        test_start = validation_end

        test_end = (
            test_start
            + test_groups
        )

        test_ids = shuffled_groups[
            test_start:test_end
        ]

        # ------------------------------------------------------
        # Build Slice Pools.
        # ------------------------------------------------------

        train_pool = self._build_pool(
            groups,
            train_ids,
        )

        validation_pool = self._build_pool(
            groups,
            validation_ids,
        )

        test_pool = self._build_pool(
            groups,
            test_ids,
        )

        # ------------------------------------------------------
        # Select exact Slice numbers.
        # ------------------------------------------------------

        train_samples = self._select_slices(
            train_pool,
            train_size,
        )

        validation_samples = self._select_slices(
            validation_pool,
            validation_size,
        )

        test_samples = self._select_slices(
            test_pool,
            test_size,
        )

        return (
            train_samples,
            validation_samples,
            test_samples,
        )

    # ==========================================================
    # 4. Filtering and Source discovery
    # ==========================================================

    def _filter_condition(
        self,
        data_list: list[BenchmarkData],
        condition: int,
    ) -> list[BenchmarkData]:
        """Filter Slices by Condition."""

        result = []

        for data in data_list:

            info = data.metadata["info"]

            if info["rpm"] != condition:
                continue

            if (
                info["fault_size"]
                in self.excluded_fault_sizes
            ):
                continue

            if info["fault_label"] == "OR":

                position = (
                    data.metadata["extra_info"]
                    ["fault_position_num"]
                )

                if position != self.or_fault_position:
                    continue

            result.append(data)

        return result

    def _discover_sources(
        self,
        data_list: list[BenchmarkData],
    ) -> list[tuple]:
        """Find all unique Sources."""

        sources = set()

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

            sources.add(source)

        return sorted(
            sources,
            key=self._source_sort_key,
        )

    def _filter_source(
        self,
        data_list: list[BenchmarkData],
        source: tuple,
    ) -> list[BenchmarkData]:
        """Return all Slices from one Source."""

        result = []

        for data in data_list:

            metadata = data.metadata

            current_source = (
                metadata["dataset"],
                metadata["info"]["rpm"],
                metadata["info"]["load"],
                metadata["info"]["fault_size"],
                metadata["info"]["file_name"],
                metadata["extra_info"]["source_key"],
            )

            if current_source == source:
                result.append(data)

        return result

    # ==========================================================
    # 5. Slice selection
    # ==========================================================

    def _build_pool(
        self,
        groups: dict,
        group_ids: list[int],
    ) -> list[BenchmarkData]:
        """Build a Slice Pool from selected Groups."""

        pool = []

        for group_id in group_ids:
            pool.extend(groups[group_id])

        return pool

    def _select_slices(
        self,
        pool: list[BenchmarkData],
        target_size: int,
    ) -> list[BenchmarkData]:
        """Randomly select exact Slice count."""

        if target_size == 0:
            return []

        if target_size > len(pool):
            raise ValueError(
                f"Requested {target_size} Slices, "
                f"but only {len(pool)} are available."
            )

        indices = self.rng.choice(
            len(pool),
            size=target_size,
            replace=False,
        )

        return [
            deepcopy(pool[index])
            for index in indices
        ]

    # ==========================================================
    # Utilities
    # ==========================================================

    def _allocate(
        self,
        total: int,
        items: list,
    ) -> list[int]:
        """Allocate quantity with earlier items receiving remainders."""

        base = total // len(items)
        remainder = total % len(items)

        return [
            base + (
                1 if index < remainder else 0
            )
            for index in range(len(items))
        ]

    def _source_sort_key(
        self,
        source: tuple,
    ) -> tuple:
        """Return a stable sorting key for Source."""

        return (
            source[0],
            source[1],
            source[2],
            "" if source[3] is None else source[3],
            source[4],
            source[5],
        )

    def _group_length(
        self,
        data: BenchmarkData,
    ) -> int:
        """Get Group length from group_info.str_end."""

        str_end = data.metadata[
            "group_info"
        ]["str_end"]

        start, end = map(
            int,
            str_end.strip("[]").split("-"),
        )

        return end - start + 1

    def _format_source(
        self,
        source: tuple,
    ) -> str:
        """Format Source information."""

        return (
            f"dataset={source[0]}, "
            f"rpm={source[1]}, "
            f"load={source[2]}, "
            f"fault_size={source[3]}, "
            f"file_name={source[4]}, "
            f"source_key={source[5]}"
        )

    # ==========================================================
    # Finalization
    # ==========================================================

    def _finalize(
        self,
        data_list: list[BenchmarkData],
    ) -> None:
        """Assign sample IDs, labels, and status."""

        for sample_id, data in enumerate(
            data_list
        ):

            data.metadata["sample_info"] = {
                "sample_id": sample_id
            }

            data.y = self._build_label(data)

            data.metadata["status"] = "sampled"

    def _build_label(
        self,
        data: BenchmarkData,
    ) -> str:
        """Build Sample Label."""

        info = data.metadata["info"]

        rpm = info["rpm"]
        fault_label = info["fault_label"]

        if fault_label == "Normal":
            return f"{rpm}-Normal"

        return (
            f"{rpm}-"
            f"{info['fault_size']}-"
            f"{fault_label}"
        )

    # ==========================================================
    # Save
    # ==========================================================

    def _save(
        self,
        train_set: list[BenchmarkData],
        validation_set: list[BenchmarkData],
        test_set: list[BenchmarkData],
    ) -> None:
        """Save train, validation, and test sets to a .pt file."""

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "train": train_set,
            "validation": validation_set,
            "test": test_set,
        }

        torch.save(
            data,
            self.output_file,
        )

    # ==========================================================
    # Leakage check
    # ==========================================================

    def _check_leakage(
        self,
        train_set: list[BenchmarkData],
        validation_set: list[BenchmarkData],
        test_set: list[BenchmarkData],
    ) -> None:
        """Check Source + Group leakage."""

        train_groups = self._group_keys(
            train_set
        )

        validation_groups = self._group_keys(
            validation_set
        )

        test_groups = self._group_keys(
            test_set
        )

        if train_groups & validation_groups:
            raise ValueError(
                "Group leakage detected between "
                "Train and Validation."
            )

        if train_groups & test_groups:
            raise ValueError(
                "Group leakage detected between "
                "Train and Test."
            )

        if validation_groups & test_groups:
            raise ValueError(
                "Group leakage detected between "
                "Validation and Test."
            )

    def _group_keys(
        self,
        data_list: list[BenchmarkData],
    ) -> set[tuple]:
        """Build Source + Group identifiers."""

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