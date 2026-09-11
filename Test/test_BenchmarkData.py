import numpy as np
from src.BenchmarkData import BenchmarkData



X = np.random.randn(10, 1024)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Create a BenchmarkData object
data = BenchmarkData(
    X=X,
    y=y,
    metadata={
        "dataset": "TestData",
        "info": {},
        "extra_info": {}
    }
)


# Print basic information
print("X shape:", data.X.shape)
print("y shape:", data.y.shape)
print("Number of samples:", data.num_samples)
print("Has labels:", data.has_labels)
print("Metadata:", data.metadata)