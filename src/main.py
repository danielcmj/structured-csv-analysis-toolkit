import os

from .dataset import ExperimentDataset
from .plotting import DataPlotter


def main():
    """
    Entry point for running the analysis pipeline.

    This script:
    - loads measurement files from a local folder
    - computes derived quantities
    - generates a set of standard diagnostic and analysis plots

    Raw data is expected to live outside the repository and is not tracked by git.
    """

    # ------------------------------------------------------------------
    # 1) Load dataset
    # ------------------------------------------------------------------
    data_folder = os.getcwd()
    ds = ExperimentDataset.import_folder(data_folder)
    n_samples = len(ds.samples)

    print(f"Loaded {n_samples} samples from '{data_folder}'")

    if n_samples == 0:
        print(
            "No measurement files found.\n"
            "Place your data in a local folder (not tracked by git)\n"
            "and update `data_folder` if needed."
        )
        return

    plotter = DataPlotter(ds)

    # ------------------------------------------------------------------
    # 2) Global diagnostics
    # ------------------------------------------------------------------
    # Overlay of all WB arms (same-bridge output)
    plotter.plot_overlay_by_position(output_type="same")

    # Bridge balance diagnostics for D7
    plotter.plot_d7_balance()

    # ------------------------------------------------------------------
    # 3) Angular analysis by die type
    # ------------------------------------------------------------------
    def run_angular_analysis(die_label, pairs, measurements=None):
        samples = ds.filter(die=die_label)
        if not samples:
            print(f"No {die_label} samples found.")
            return

        sub_ds = ExperimentDataset()
        for s in samples:
            sub_ds.add(s)
        sub_ds.compute_WB_outputs()

        sub_plotter = DataPlotter(sub_ds)

        if measurements is not None:
            sub_plotter.plot_overlay_by_position(
                output_type="same",
                measurements=measurements,
            )

        for p in pairs:
            sub_plotter.plot_ratio_and_angle(p, output_type="same")

    # A1
    a1_pairs = [(1, 3), (2, 8), (5, 7), (4, 6)]
    run_angular_analysis("A1", a1_pairs)

    # D6
    d6_pairs = [(1, 3), (2, 4)]
    run_angular_analysis(
        "D6",
        d6_pairs,
        measurements=[f"WB{i}" for i in range(1, 5)],
    )

    # C6
    c6_pairs = [(1, 3), (2, 4)]
    run_angular_analysis(
        "C6",
        c6_pairs,
        measurements=[f"WB{i}" for i in range(1, 5)],
    )

    # ------------------------------------------------------------------
    # 4) MR curves vs fluence (D7)
    # ------------------------------------------------------------------
    fluences = [65, 60, 55, 50, 45, 40, 35, 30, 25]

    plotter.plot_d7_mr_curves(fluences, mode="split_chiplet")
    plotter.plot_d7_mr_curves(fluences, mode="same_chiplet_die")
    plotter.plot_d7_mr_curves(fluences, mode="merged")
    plotter.plot_d7_mr_curves(fluences, mode="compare_rbx", rb_label="RB2")


if __name__ == "__main__":
    main()