import os
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler

@dataclass(frozen=True)
class SampleMetadata:
    wafer: str
    anneal: str
    die:   str
    x_coord: int
    y_coord: int
    coord_type: str
    measurement: str
    direction: str

    @staticmethod
    def from_filename(filepath: str) -> 'SampleMetadata':
        name = os.path.splitext(os.path.basename(filepath))[0]
        parts = name.split('_')
        if len(parts) < 4:
            raise ValueError(f"Filename '{name}' does not match expected pattern.")
        wafer, anneal, coord_str, meas = parts[:4]

        
        try:
            x = int(coord_str[1:3])
            y = int(coord_str.split('y')[1])
        except:
            raise ValueError(f"Cannot parse coordinates from '{coord_str}'")
        coord_map = {'19': 'A1', '00': 'D5'}
        coord_type = coord_map.get(f"{y:02d}", f"y{y:02d}")

        # direction logic…
        if meas.upper().startswith('RB'):
            direction = 'CW'
        else:
            direction = parts[4] if len(parts) >= 5 else None
            if direction is None:
                raise ValueError(f"Missing direction for '{meas}' in '{name}'")

        # the die folder is the **immediate** parent of the file
        die = os.path.basename(os.path.dirname(filepath))

        return SampleMetadata(
            wafer=wafer,
            anneal=anneal,
            die=die,
            x_coord=x,
            y_coord=y,
            coord_type=coord_type,
            measurement=meas,
            direction=direction
        )

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

class ExperimentDataset:
    def __init__(self):
        self.samples = []

    def add(self, sample: MeasurementData):
        self.samples.append(sample)

    def import_all(self, folder: str, skiprows: int = 8) -> 'ExperimentDataset':
        """Import all .txt files under `folder` into the dataset."""
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.txt'):
                    self.add(MeasurementData(
                        os.path.join(root, f), skiprows
                    ))
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

class DataPlotter:
    """Plotting and analysis tools built on an ExperimentDataset."""
    def __init__(self, dataset: ExperimentDataset):
        self.dataset = dataset
        
    def plot_pair_cw(
        self,
        pair: tuple = ('WB1','WB2'),
        output_type: str = 'same'
    ):
        """
        For CW only, overlay two WB traces (e.g. WB1 vs WB2) and
        plot angle vs. bridge‐output (same or crossed).
        """
        col = 'WB_out_same' if output_type=='same' else 'WB_out_crossed'

        # group by wafer+anneal
        groups = defaultdict(list)
        for s in self.dataset.samples:
            if not s.metadata.measurement.upper().startswith('WB'):
                continue
            if col not in s.data:
                continue
            groups[(s.metadata.wafer, s.metadata.anneal)].append(s)

        for (wafer,anneal), samples in groups.items():
            # pick out only the CW sweeps
            cw = [s for s in samples if s.metadata.direction=='CW']
            if not cw:
                continue

            plt.figure(figsize=(6,4))
            for meas in pair:
                # find the sample matching this label
                s = next((s for s in cw if s.metadata.measurement.upper()==meas), None)
                if s is None:
                    print(f"  → no {meas} in {wafer}_{anneal}_CW, skipping")
                    continue

                # drop NaNs, sort by angle
                df = s.data.dropna(subset=['angle',col]).copy()
                df = df.sort_values('angle')

                plt.plot(df['angle'], df[col], 
                         marker='o', linestyle='-', label=meas)

            plt.title(f"{wafer}_{anneal}_CW_{col}")
            plt.xlabel('Angle (°)')
            plt.ylabel(f"{col} (V)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
    def plot_overlay_by_position(
        self,
        output_type: str = 'same',
        measurements: list = None
    ):
        """
        Overlay a custom list of WB arms (e.g. ['WB1','WB2',…,'WB8'])
        for each wafer+anneal+coord, split CW/CCW, over 0→720°.
        """
        col = 'WB_out_same' if output_type == 'same' else 'WB_out_crossed'

        if measurements is None:
            measurements = [f'WB{i}' for i in range(1,9)]

        # group by wafer, anneal, x, y
        pos_groups = defaultdict(list)
        for s in self.dataset.samples:
            if not s.metadata.measurement.upper().startswith('WB'):
                continue
            if col not in s.data.columns:
                continue
            key = (
                s.metadata.wafer,
                s.metadata.anneal,
                s.metadata.x_coord,
                s.metadata.y_coord
            )
            pos_groups[key].append(s)

        for (wafer, anneal, x, y), samples in pos_groups.items():
            for direction in ('CW','CCW'):
                dir_samples = [s for s in samples if s.metadata.direction == direction]
                if not dir_samples:
                    continue

                # create a fresh figure & axes
                fig, ax = plt.subplots(figsize=(8,4))

                # apply a vivid tab10 cycle to *this* axes
                vivid = plt.get_cmap('tab10').colors
                ax.set_prop_cycle(cycler('color', vivid))

                for meas_label in measurements:
                    s = next(
                        (s for s in dir_samples
                         if s.metadata.measurement.upper() == meas_label),
                        None
                    )
                    if s is None:
                        continue

                    df = s.data.dropna(subset=['angle', col]).sort_values('angle')
                    ax.plot(
                        df['angle'],
                        df[col],
                        linestyle='-',  # line‐only
                        label=meas_label
                    )

                title = f"{wafer}_{anneal}_x{x:02d}y{y:02d}_{direction}_{col}"
                ax.set_title(title)
                ax.set_xlabel('Angle (°)')
                ax.set_ylabel(f"{col} (V)")
                ax.legend(title=direction, ncol=2)
                ax.grid(True)
                plt.tight_layout()
                plt.show()


# =============================================================================
# Methods for each die
# =============================================================================
    def plot_d7_balance(self):
        """
        D7: show how close each bridge arm (DMM1 vs. DMM2) is to zero.
        Produces two figures:
          1) Overlay of (DMM1–DMM2) vs. angle
          2) Raincloud with big centred reminder that violin width ∝ frequency
        """
        # 1) select only the D7‐die samples
        d7 = [s for s in self.dataset.samples if s.metadata.die == 'D7']
        if not d7:
            print("No D7 samples found.")
            return

        # ── FIGURE 1: imbalance vs. angle overlays ──
        fig1, ax1 = plt.subplots(figsize=(8,5))
        labels1 = []
        for s in d7:
            df = s.data.dropna(subset=['angle','DMM1','DMM2'])
            imb = df['DMM1'] - df['DMM2']
            lbl = f"{s.metadata.wafer}_{s.metadata.anneal}_{s.metadata.measurement}_{s.metadata.direction}"
            labels1.append(lbl)
            ax1.plot(df['angle'], imb, alpha=0.6)
        ax1.axhline(0, color='k', linestyle='--', linewidth=1)
        ax1.set_title("D7 Die — Bridge Imbalance vs. Angle", fontsize=14)
        ax1.set_xlabel("Angle (deg)", fontsize=12)
        ax1.set_ylabel("DMM1 − DMM2 (V)", fontsize=12)
        ax1.grid(True)
        ax1.text(
            0.95, 0.95, r"$V_{\!bias}=1\,\mathrm{V}$",
            transform=ax1.transAxes,
            fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        ax1.legend(labels1, loc='center left', bbox_to_anchor=(1,0.5), fontsize=10)
        plt.setp(ax1.get_xticklabels(), fontsize=10)
        plt.setp(ax1.get_yticklabels(), fontsize=10)
        plt.tight_layout()
        plt.show()

        # ── FIGURE 2: raincloud distribution ──
        imbs = []; labels2 = []
        for s in d7:
            df = s.data.dropna(subset=['DMM1','DMM2'])
            arr = (df['DMM1'] - df['DMM2']).values
            if arr.size:
                imbs.append(arr)
                labels2.append(f"{s.metadata.wafer}_{s.metadata.anneal}_{s.metadata.measurement}_{s.metadata.direction}")

        fig2, ax2 = plt.subplots(figsize=(10,4))

        # 2a) violin
        parts = ax2.violinplot(imbs, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)

        # 2b) slim boxplot
        pos = np.arange(1, len(imbs)+1)
        ax2.boxplot(
            imbs, positions=pos, widths=0.1, showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
            medianprops=dict(color='black', linewidth=2)
        )

        # 2c) jittered raw points
        for i, arr in enumerate(imbs, start=1):
            x = np.random.normal(i, 0.04, size=arr.size)
            ax2.scatter(x, arr, color='black', s=5, alpha=0.3)

        # 2d) mean ± std
        means = [np.mean(a) for a in imbs]
        stds  = [np.std(a)  for a in imbs]
        ax2.errorbar(pos, means, yerr=stds, fmt='o', color='k', label='mean ± 1σ')

        ax2.set_xticks(pos)
        ax2.set_xticklabels(labels2, rotation=45, ha='right', fontsize=12)
        ax2.set_ylabel("DMM1 − DMM2 (V)", fontsize=12)
        ax2.set_title("D7 Die — Bridge Imbalance Distribution by Sample", fontsize=14)
        ax2.grid(True, axis='y')
        plt.setp(ax2.get_yticklabels(), fontsize=10)

        # bias‐voltage textbox
        ax2.text(
            0.95, 0.95, r"$V_{\!bias}=1\,\mathrm{V}$",
            transform=ax2.transAxes,
            fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        # centered violin‐width note, large font\
        ax2.text(
            0.5, 0.95,
            r"Violin width $\propto$ frequency of $\Delta V = \mathrm{DMM1}-\mathrm{DMM2}$",
            transform=ax2.transAxes,
            fontsize=14,
            va='top', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        ax2.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.show()


    def plot_ratio_and_angle(
        self,
        pair: tuple = ('WB2', 'WB6'),
        output_type: str = 'same'
    ):
        """
        For each wafer+anneal+direction+coordinate:
          - ratio = WB_out_{same|crossed} of the two legs in `pair`
          - computed_angle = arctan2(num, den), unwrapped & shifted, deg
          - diff = angle - computed_angle
        Plots: ratio vs angle, angle vs angle, diff vs angle.
        """
        # allow numeric pairs
        num_key, den_key = pair
        if isinstance(num_key, int):
            num_key = f"WB{num_key}"
            den_key = f"WB{den_key}"
        pair = (num_key, den_key)

        out_col = 'WB_out_same' if output_type == 'same' else 'WB_out_crossed'

        # group into quads
        quad_groups = defaultdict(list)
        for s in self.dataset.samples:
            key = (
                s.metadata.wafer,
                s.metadata.anneal,
                s.metadata.x_coord,
                s.metadata.y_coord,
                s.metadata.direction
            )
            quad_groups[key].append(s)

        for (wafer, anneal, x, y, direction), samples in quad_groups.items():
            if direction != 'CW':
                continue

            # pick out the two dataframes
            data   = {s.metadata.measurement.upper(): s.data for s in samples if out_col in s.data}
            num_df = data.get(pair[0])
            den_df = data.get(pair[1])
            if num_df is None or den_df is None:
                continue

            # align on angle
            df_num = num_df[['angle', out_col]].rename(columns={out_col: 'num'})
            df_den = den_df[['angle', out_col]].rename(columns={out_col: 'den'})
            dfm    = pd.merge(df_num, df_den, on='angle', how='inner').dropna(subset=['num','den'])
            if dfm.empty:
                continue

            angles = dfm['angle'].values
            ratio  = dfm['num'].values / dfm['den'].values

            # arctan2 → unwrap → shift zero → degrees
            raw       = np.arctan2(dfm['num'].values, dfm['den'].values)
            unwrapped = np.unwrap(raw) - raw[~np.isnan(raw)][0]
            comp      = np.degrees(unwrapped)

            # force positive slope
            slope = np.polyfit(np.degrees(np.unwrap(np.radians(angles))), comp, 1)[0]
            if slope < 0:
                comp = -comp

            diff = angles - comp
            sample_id = f"{wafer}_{anneal}_x{x:02d}y{y:02d}_{direction}"

            # — Ratio vs Angle —
            plt.figure(figsize=(6,4))
            plt.plot(angles, ratio, 'o-', label=f"{pair[0]}/{pair[1]} ({output_type})")
            plt.title(f"Ratio {pair[0]}/{pair[1]} ({output_type}) — {sample_id}")
            plt.xlabel('Angle (deg)')
            plt.ylabel('Ratio')
            plt.grid(True); plt.legend(); plt.show()

            # — Computed angle vs Angle —
            plt.figure(figsize=(6,4))
            plt.plot(angles, comp, 's-', label=f"Computed angle ({output_type})", ms=3)
            plt.plot([angles.min(), angles.max()], [angles.min(), angles.max()],
                     '--', label='x=y')
            plt.title(f"Angle {pair[0]}/{pair[1]} ({output_type}) — {sample_id}")
            plt.xlabel('Angle (deg)')
            plt.ylabel('Computed angle (deg)')
            plt.grid(True); plt.legend(); plt.show()

            # — Difference vs Angle —
            plt.figure(figsize=(6,4))
            plt.plot(angles, diff, 'o-', label=f"Diff ({output_type})")
            plt.title(f"Diff {pair[0]}/{pair[1]} ({output_type}) — {sample_id}")
            plt.xlabel('Angle (deg)')
            plt.ylabel('Diff (deg)')
            plt.grid(True); plt.legend(); plt.show()
  
    
    def plot_d7_mr_curves(self, fluence_map, mode='split_chiplet', rb_label='RB1'):
        """
        Plot R vs angle for D7 die across fluence levels, with flexible plotting modes.
    
        Parameters
        ----------
        fluence_map : list
            List mapping x_coord = 1,2,... to fluence values (e.g. [65, 60, ...]).
        mode : str
            One of:
            - 'split_chiplet'   → one plot per chiplet (wafer+anneal)
            - 'same_chiplet_die'→ one plot per DIE within chiplet
            - 'merged'          → all RB arms together in one figure
            - 'compare_rbx'     → only RBx across fluences
        rb_label : str
            Used only if mode == 'compare_rbx'; filters to RB1, RB2, etc.
        """
        d7 = [s for s in self.dataset.samples if s.metadata.die == 'D7']
        if not d7:
            print("No D7 samples found.")
            return
    
        # Group samples by chiplet
        chiplet_groups = defaultdict(list)
        for s in d7:
            if not s.metadata.measurement.upper().startswith('RB'):
                continue
            chiplet = f"{s.metadata.wafer}_{s.metadata.anneal}"
            chiplet_groups[chiplet].append(s)
    
        def compute_curve(s):
            df = s.data.dropna(subset=['angle', 'smu_curr', 'smu_volt']).copy()
            if df.empty:
                return None
            df = df.groupby('angle', as_index=False).mean()
            df['R'] = df['smu_volt'] / df['smu_curr'].replace(0, np.nan)
            return df
    
        # Handle different modes
        if mode == 'split_chiplet':
            for chiplet, samples in chiplet_groups.items():
                fig, ax = plt.subplots(figsize=(8, 4))
                x_groups = defaultdict(list)
                for s in samples:
                    x_groups[s.metadata.x_coord].append(s)
    
                for x in sorted(x_groups):
                    group = x_groups[x]
                    fluence = fluence_map[x - 1] if x - 1 < len(fluence_map) else None
                    flu_str = f"{fluence}%" if fluence is not None else "unknown"
                    for s in group:
                        df = compute_curve(s)
                        if df is not None:
                            label = f"{s.metadata.measurement.upper()} ({flu_str})"
                            ax.plot(df['angle'], df['R'], label=label)
    
                ax.set_title(f"D7 curves — {chiplet}", fontsize=14)
                ax.set_xlabel("Angle (°)")
                ax.set_ylabel("R ($\\Omega$)")
                ax.grid(True)
                ax.legend(title='RB / Fluence', fontsize=10)
                plt.tight_layout()
                plt.show()
    
        elif mode == 'same_chiplet_die':
            for chiplet, samples in chiplet_groups.items():
                die_groups = defaultdict(list)
                for s in samples:
                    die_groups[s.metadata.x_coord].append(s)
                for x in sorted(die_groups):
                    fluence = fluence_map[x - 1] if x - 1 < len(fluence_map) else None
                    flu_str = f"{fluence}%" if fluence is not None else "unknown"
    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    for s in die_groups[x]:
                        df = compute_curve(s)
                        if df is not None:
                            label = f"{s.metadata.measurement.upper()} ({flu_str})"
                            ax.plot(df['angle'], df['R'], label=label)
    
                    ax.set_title(f"{chiplet} — x={x} (fluence {flu_str})", fontsize=14)
                    ax.set_xlabel("Angle (°)")
                    ax.set_ylabel("R ($\\Omega$)")
                    ax.grid(True)
                    ax.legend(fontsize=10)
                    plt.tight_layout()
                    plt.show()
    
        elif mode == 'merged':
            fig, ax = plt.subplots(figsize=(10, 5))
        
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(fluence_map), max(fluence_map))
        
            for chiplet, samples in chiplet_groups.items():
                for s in samples:
                    df = compute_curve(s)
                    if df is None:
                        continue
                    x = s.metadata.x_coord
                    if x - 1 < len(fluence_map):
                        fluence = fluence_map[x - 1]
                        flu_str = f"{fluence}%"
                        color = cmap(norm(fluence))
                    else:
                        fluence = None
                        flu_str = "unknown"
                        color = "gray"
        
                    label = f"{s.metadata.measurement.upper()} ({flu_str}, {chiplet})"
                    ax.plot(df['angle'], df['R'], label=label, color=color, alpha=0.8)
        
            ax.set_title("All D7 R vs angle (merged)", fontsize=14)
            ax.set_xlabel("Angle (°)")
            ax.set_ylabel("R ($\\Omega$)")
            ax.grid(True)
        
            # Colorbar for fluence
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Laser Power (%)")
        
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.show()

    
        elif mode == 'compare_rbx':
            rb_label = rb_label.upper()
            fig, ax = plt.subplots(figsize=(8, 4))
            for chiplet, samples in chiplet_groups.items():
                rbs = [s for s in samples if s.metadata.measurement.upper() == rb_label]
                for s in rbs:
                    df = compute_curve(s)
                    if df is None:
                        continue
                    x = s.metadata.x_coord
                    fluence = fluence_map[x - 1] if x - 1 < len(fluence_map) else None
                    flu_str = f"{fluence}%" if fluence is not None else "unknown"
                    label = f"{chiplet} x{x} ({flu_str})"
                    ax.plot(df['angle'], df['R'], label=label)
    
            ax.set_title(f"{rb_label} vs angle across fluences", fontsize=14)
            ax.set_xlabel("Angle (°)")
            ax.set_ylabel("R ($\\Omega$)")
            ax.grid(True)
            ax.legend(fontsize=9)
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError(f"Unknown mode: '{mode}'")





def debug_reading(ds, max_samples=5, n_raw=10, n_df=5):
    """
    For up to `max_samples` MeasurementData in `ds`,
    print the first `n_raw` raw lines of the file and then
    the first `n_df` rows of the parsed DataFrame.
    """
    print("=== DEBUGGING FILE READ / PARSE ===")
    for i, sample in enumerate(ds.samples):
        if i >= max_samples:
            break
        print(f"\n---- Sample {i+1}: {sample.filepath} ----")
        # 1) raw lines
        print(" raw file lines:")
        with open(sample.filepath, 'r') as f:
            for _ in range(n_raw):
                line = f.readline()
                if not line:
                    break
                print("   ", line.rstrip())
        # 2) DataFrame head
        print(" parsed DataFrame (first rows):")
        print(sample.data.head(n_df).to_string().replace("\n", "\n   "))
        # 3) Check the angle column monotonicity
        ang = sample.data['angle'].values
        diffs = np.diff(ang)
        print(f"   angle diffs (first 5): {diffs[:5]}\n"
              f"   any negative diffs? {bool((diffs<0).any())}")
    print("=== END DEBUG ===\n")
            
# =============================================================================
# if __name__ == '__main__':
#     folder = os.getcwd()
#     ds = ExperimentDataset.import_folder(folder)
#     print(f"Loaded {len(ds.samples)} samples from '{folder}'")
#     ds.print_files_by_wafer()
# 
#     # filter only WXXXX_YY
#     sub = [s for s in ds.samples if s.metadata.wafer=='W9122' and s.metadata.anneal=='C3' and s.metadata.die =='D6']
#     sub_ds = ExperimentDataset()
#     for s in sub:
#         sub_ds.add(s)
#     sub_ds.compute_WB_outputs()
# 
#     plotter = DataPlotter(sub_ds)
#     # overlay plots
#     plotter.plot_overlay_by_wafer_anneal('same')
#     plotter.plot_overlay_by_wafer_anneal('crossed')
# =============================================================================

# =============================================================================
#     pairs = [('WB2','WB4'), ('WB6','WB8')]
#     for out_type in ['same', 'crossed']:
#         for pair in pairs:
#             plotter.plot_ratio_and_angle(pair, output_type=out_type)
# 
# =============================================================================


# =============================================================================
# # PRINT DATA FOR BRIDGES SAME AND CROSSED
# ################
# ################
# 
# if __name__ == '__main__':
# 
#     # 1) load everything
#     folder = os.getcwd()
#     ds = ExperimentDataset.import_folder(folder)
#     print(f"Loaded {len(ds.samples)} samples from '{folder}'")
# 
#     plotter = DataPlotter(ds)
# 
#     # 2) overlay all WB-outs (same vs crossed)
#     plotter.plot_overlay_by_wafer_anneal('same')
#     plotter.plot_overlay_by_wafer_anneal('crossed')
# 
#     # 3) ratio/angle for the four bridge-pairs you specified
#     pairs = [
#         ('WB1','WB5'),
#         ('WB2','WB6'),
#         ('WB3','WB7'),
#         ('WB4','WB8'),
#     ]
#     for num, den in pairs:
#         plotter.plot_ratio_and_angle((num,den), output_type='same')
#         plotter.plot_ratio_and_angle((num,den), output_type='crossed')
# =============================================================================


if __name__ == '__main__':

    # 0) load your full dataset
    folder = os.getcwd()
    ds     = ExperimentDataset.import_folder(folder)
    print(f"Loaded {len(ds.samples)} samples from '{folder}'")

    plotter = DataPlotter(ds)

    # 1) Global overlay (all eight arms, same only)
    plotter.plot_overlay_by_position('same')

    # 2) D7 balance
    plotter.plot_d7_balance()

    # 3) A1 angular (pairs 1,3 2,4 5,7 6,8 — same + crossed)
    #a1_pairs   = [(1, 3), (2, 4), (5, 7), (6, 8)]
    a1_pairs   = [(1, 3), (2, 8), (5, 7), (4, 6)]
    a1_samples = ds.filter(die='A1')
    if not a1_samples:
        print("No A1 samples found.")
    else:
        a1_ds      = ExperimentDataset()
        for s in a1_samples:
            a1_ds.add(s)
        a1_ds.compute_WB_outputs()
        a1_plotter = DataPlotter(a1_ds)

        for p in a1_pairs:
            a1_plotter.plot_ratio_and_angle(p, output_type='same')
            #a1_plotter.plot_ratio_and_angle(p, output_type='crossed')

    # 4) D6 angular (pairs 1,3 & 2,4 — only WB1–WB4, same only)
    d6_pairs   = [(1, 3), (2, 4)]
    d6_samples = ds.filter(die='D6')
    if not d6_samples:
        print("No D6 samples found.")
    else:
        d6_ds      = ExperimentDataset()
        for s in d6_samples:
            d6_ds.add(s)
        d6_ds.compute_WB_outputs()
        d6_plotter = DataPlotter(d6_ds)

        # overlay only WB1–4
        d6_plotter.plot_overlay_by_position(
            'same',
            measurements=[f'WB{i}' for i in range(1, 5)]
        )
        # ratio/angle for the two pairs (same only)
        for p in d6_pairs:
            d6_plotter.plot_ratio_and_angle(p, output_type='same')

    # 5) C6 angular (same structure as D6: only WB1–WB4, pairs 1,3 & 2,4)
    c6_pairs   = [(1, 3), (2, 4)]
    c6_samples = ds.filter(die='C6')
    if not c6_samples:
        print("No C6 samples found.")
    else:
        c6_ds      = ExperimentDataset()
        for s in c6_samples:
            c6_ds.add(s)
        c6_ds.compute_WB_outputs()
        c6_plotter = DataPlotter(c6_ds)

        # overlay only WB1–4
        c6_plotter.plot_overlay_by_position(
            'same',
            measurements=[f'WB{i}' for i in range(1, 5)]
        )
        # ratio/angle for the two pairs (same only)
        for p in c6_pairs:
            c6_plotter.plot_ratio_and_angle(p, output_type='same')

    
    fluences = [65, 60, 55, 50, 45, 40, 35, 30, 25]

    # 1. Current style (split by chiplet)
    plotter.plot_d7_mr_curves(fluences, mode='split_chiplet')
    
    # 2. All RBs of each die (i.e., x) in one plot
    plotter.plot_d7_mr_curves(fluences, mode='same_chiplet_die')
    
    # 3. All data merged in one giant plot
    plotter.plot_d7_mr_curves(fluences, mode='merged')
    
    # 4. Compare RB1 across fluences
    plotter.plot_d7_mr_curves(fluences, mode='compare_rbx', rb_label='RB2')




# =============================================================================
# ### Example usage for plotting pairs one by one
# if __name__=='__main__':
#     import os
# 
#     folder = os.getcwd()
#     ds = ExperimentDataset.import_folder(folder)
# 
#     # make a dataset containing *only* CW traces:
#     cw_ds = ExperimentDataset()
#     for s in ds.samples:
#         if s.metadata.direction=='CW':
#             cw_ds.add(s)
#     cw_ds.compute_WB_outputs()
# 
#     plotter = DataPlotter(cw_ds)
#     # now plot WB1 vs WB2 on the same axes,
#     # using the 'same' bridge output
#     plotter.plot_pair_cw(('WB1','WB2'), output_type='same')
#     plotter.plot_pair_cw(('WB1','WB3'), output_type='same')
#     plotter.plot_pair_cw(('WB1','WB4'), output_type='same')
# =============================================================================


# =============================================================================
# if __name__=='__main__':
#     folder = os.getcwd()
#     ds = ExperimentDataset.import_folder(folder)
#     plotter = DataPlotter(ds)
# 
#     # plot *all* eight arms:
#     plotter.plot_overlay_by_position('same')
# 
#     # or, explicitly:
#     arms = [f'WB{i}' for i in range(1,9)]
#     #plotter.plot_overlay_by_position('crossed', measurements=arms)
# =============================================================================
