from .dataset import ExperimentDataset

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
