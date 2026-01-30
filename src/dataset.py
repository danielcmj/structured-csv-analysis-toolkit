from .loader import MeasurementData
import os

class ExperimentDataset:
    def __init__(self):
        self.samples = []

    def add(self, sample: MeasurementData):
        self.samples.append(sample)

    def import_all(self, folder: str, skiprows: int = 8) -> 'ExperimentDataset':
        """Import all .txt files under `folder` into the dataset."""
        for root, _, files in os.walk(folder):
            for f in files:
                  name = os.path.splitext(f)[0]
                  # only load measurement files that start with wafer id (e.g., W9122_...)
                  if f.lower().endswith(".txt") and name.upper().startswith("W"):
                        self.add(MeasurementData(os.path.join(root, f), skiprows))
            self.compute_WB_outputs()
        return self

    @classmethod
    def import_folder(cls, folder: str, skiprows: int = 8) -> 'ExperimentDataset':
        """Convenience constructor: load and compute all outputs."""
        return cls().import_all(folder, skiprows)

    def compute_WB_outputs(self):
        """Compute WB_out_same and WB_out_crossed for all WB samples."""
        # same output
        for s in self.samples:
            if s.metadata.measurement.upper().startswith('WB'):
                s.data['WB_out_same'] = s.data['DMM1'] - s.data['DMM2']

        # build an index so we can look up partners
        idx = {
            (
                s.metadata.wafer,
                s.metadata.anneal,
                s.metadata.x_coord,
                s.metadata.y_coord,
                s.metadata.direction,
                s.metadata.measurement.upper()
            ): s
            for s in self.samples
        }

        pairs = {'WB2': 'WB6', 'WB6': 'WB2', 'WB4': 'WB8', 'WB8': 'WB4'}
        for s in self.samples:
            key = (
                s.metadata.wafer,
                s.metadata.anneal,
                s.metadata.x_coord,
                s.metadata.y_coord,
                s.metadata.direction,
                s.metadata.measurement.upper()
            )
            meas = key[-1]
            if meas in pairs:
                partner = idx.get(key[:-1] + (pairs[meas],))
                # crossed = DMM1(this) minus DMM2(partner)
                s.data['WB_out_crossed'] = (
                    s.data['DMM1'] - partner.data['DMM2']
                ) if partner else np.nan

    def group_by(self, attr: str) -> dict:
        """Group samples by metadata attribute (e.g. 'wafer', 'anneal', etc.)."""
        groups = defaultdict(list)
        for s in self.samples:
            groups[getattr(s.metadata, attr)].append(s)
        return groups

    def filter(self, **criteria) -> list:
        """Filter samples by metadata fields (e.g. measurement='WB2', direction='CW')."""
        out = self.samples
        for attr, val in criteria.items():
            if callable(val):
                out = [s for s in out if val(getattr(s.metadata, attr))]
            else:
                out = [s for s in out if getattr(s.metadata, attr) == val]
        return out

    def print_files_by_wafer(self):
        """Print all sample file paths organized by wafer."""
        for wafer, samples in self.group_by('wafer').items():
            print(f"Wafer {wafer}:")
            for s in samples:
                print(f"  {s.filepath}")
            print()
