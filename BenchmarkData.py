from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class BenchmarkData:
    """
    Benchmark 中各模块之间传递数据的统一容器。

    Attributes
    ----------
    X : np.ndarray
        样本数据。

    y : Optional[np.ndarray]
        样本标签。
        无标签数据允许为 None，例如未来 DA 中的 target unlabeled data。

    metadata : dict
        数据的整体描述信息，包含：
        - dataset
        - info
        - extra_info
    """

    X: np.ndarray
    y: Optional[np.ndarray] = None

    metadata: dict = field(
        default_factory=lambda: {
            "dataset": None,
            "info": {},
            "extra_info": {}
        }
    )

    def __post_init__(self):
        """初始化后进行基础数据检查。"""

        # X 必须是 numpy array
        if not isinstance(self.X, np.ndarray):
            raise TypeError(
                "X must be a numpy.ndarray."
            )

        # X 至少应该有一个样本维度
        if self.X.ndim < 1:
            raise ValueError(
                "X must contain at least one dimension."
            )

        # 如果存在标签，则检查样本数量
        if self.y is not None:

            if not isinstance(self.y, np.ndarray):
                raise TypeError(
                    "y must be a numpy.ndarray or None."
                )

            if len(self.X) != len(self.y):
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got len(X)={len(self.X)}, "
                    f"len(y)={len(self.y)}."
                )

        # metadata 必须是字典
        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        # 保证三个基础字段存在
        self.metadata.setdefault("dataset", None)
        self.metadata.setdefault("info", {})
        self.metadata.setdefault("extra_info", {})

    @property
    def num_samples(self) -> int:
        """返回样本数量。"""
        return len(self.X)

    @property
    def has_labels(self) -> bool:
        """判断当前数据是否包含标签。"""
        return self.y is not None