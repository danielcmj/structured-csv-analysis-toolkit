import numpy as np

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
            