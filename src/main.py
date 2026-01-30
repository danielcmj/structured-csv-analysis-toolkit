from .metadata import SampleMetadata
from .loader import MeasurementData
from .dataset import ExperimentDataset
from .plotting import DataPlotter
from .utils import debug_reading

if __name__ == "__main__":
    
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
