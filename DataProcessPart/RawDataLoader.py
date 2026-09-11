from DataProcessPart.BenchmarkData import BenchmarkData


class RawDataLoader:
    """Unified interface for raw data loading."""

    def __init__(self, loader):
        """Initialize the raw data loader."""

        self.loader = loader

    def load(self) -> list[BenchmarkData]:
        """Load raw data using the selected dataset loader."""

        return self.loader.load()