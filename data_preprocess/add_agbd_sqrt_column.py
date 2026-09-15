# Adds an `agbd_sqrt` column to an existing patch-metadata CSV.
#
# Why this exists as a separate script rather than a build_patches change:
# build_patches_version_6 takes days to run over ~220k shots, and the sqrt target
# is derived purely from agbd_center, which that CSV already contains. So the new
# target can be appended in seconds without touching the patches themselves.
#
# Why sqrt instead of log1p:
# the model is trained on a transform and evaluated in Mg/ha, so the back-transform's
# slope decides how much a small mistake costs. For log1p the back-transform is
# expm1, whose slope is ~y itself, so the penalty scales WITH biomass and the dense
# forest shots blow up. For sqrt the back-transform is t**2, slope 2*sqrt(y), so the
# same slip costs far less at the high end (at 1000 Mg/ha: ~100 vs ~6 Mg/ha).
#
# agbd_center is strictly positive in this dataset (min 0.315), so sqrt is safe,
# but it is clipped at 0 anyway to stay robust if the CSV is ever regenerated.

import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_6.csv"

def main():
    path = Path(CSV_PATH)
    df = pd.read_csv(path)

    if "agbd_center" not in df.columns:
        raise ValueError(f"agbd_center not found in {path}")

    df["agbd_sqrt"] = np.sqrt(df["agbd_center"].clip(lower=0.0))

    print(f"rows            : {len(df)}")
    print(f"agbd_center     : {df['agbd_center'].min():.3f} .. {df['agbd_center'].max():.3f}")
    print(f"agbd_sqrt (new) : {df['agbd_sqrt'].min():.3f} .. {df['agbd_sqrt'].max():.3f}")
    if "agbd_log" in df.columns:
        print(f"agbd_log (kept) : {df['agbd_log'].min():.3f} .. {df['agbd_log'].max():.3f}")

    df.to_csv(path, index=False)
    print(f"\nwrote {path}")

if __name__ == "__main__":
    main()
