from .metadata import SampleMetadata
import pandas as pd

class MeasurementData:
    def __init__(self, filepath: str, skiprows: int = 8):
        self.filepath = filepath
        self.skiprows = skiprows
        self.metadata = SampleMetadata.from_filename(filepath)
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.filepath,
            skiprows=self.skiprows,
            sep=r",\s*",
            engine='python',
            names=['angle', 'smu_curr', 'smu_volt', 'DMM1', 'DMM2'],
            dtype=str
        )
        for c in ['angle', 'smu_volt', 'smu_curr', 'DMM1', 'DMM2']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
