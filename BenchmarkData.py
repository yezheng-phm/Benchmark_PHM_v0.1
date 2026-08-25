from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BenchmarkData:
    """
    A standard data container used between Benchmark modules.

    Attributes
    ----------
    X : np.ndarray
        Sample data.

    y : Optional[np.ndarray]
        Sample labels.
        It can be None for unlabeled data.

    metadata : dict
        General information about the data.
    """

    X: np.ndarray
    # this is for other options, for DA\DG the samples do not have labels mayde.
    y: Optional[np.ndarray] = None       

    metadata: dict = field(
        default_factory=lambda: {
            "dataset": None,
            "info": {},
            "extra_info": {}
        }
    )

    def __post_init__(self):
        """Check the data after initialization."""

        # X must be a NumPy array
        if not isinstance(self.X, np.ndarray):
            raise TypeError(
                "X must be a numpy.ndarray."
            )

        # X must have at least one dimension
        if self.X.ndim < 1:
            raise ValueError(
                "X must contain at least one dimension."
            )

        # Check y if labels are provided
        if self.y is not None:

            # y must be a NumPy array
            if not isinstance(self.y, np.ndarray):
                raise TypeError(
                    "y must be a numpy.ndarray or None."
                )

            # X and y must have the same number of samples
            if len(self.X) != len(self.y):
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got len(X)={len(self.X)}, "
                    f"len(y)={len(self.y)}."
                )

        # metadata must be a dictionary
        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        # Add default metadata fields if they are missing
        self.metadata.setdefault("dataset", None)
        self.metadata.setdefault("info", {})
        self.metadata.setdefault("extra_info", {})

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    @property
    def has_labels(self) -> bool:
        """Check if the data has labels."""
        return self.y is not None