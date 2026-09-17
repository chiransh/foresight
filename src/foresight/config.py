"""Shared config for the baseline models so their metrics are comparable.

SAMPLE_STORES is a fixed set of 12 stores stratified across all four
StoreType values and all three Assortment levels (random_state=42 over
store.csv, 3 per type). Fitting all three model families against every one
of the 1,115 stores isn't necessary to compare them honestly, but fitting
each against a different subset would be: this list is what Prophet,
LightGBM, and NHITS all train and evaluate against, so the head-to-head
comparison is over the same stores rather than different samples. Pass
--stores all to any baseline or to the backtest to run the full store set.
"""

from pathlib import Path

import pandas as pd

SAMPLE_STORES = [85, 195, 259, 347, 353, 464, 772, 791, 982, 1006, 1053, 1070]

HOLDOUT_DAYS = 42


def all_store_ids(data_dir: Path = Path("data/raw"), train_filename: str = "train.csv") -> list[int]:
    """Every store present in the data, for runs that are not using the sample."""
    stores = pd.read_csv(data_dir / train_filename, usecols=["Store"])["Store"].unique()
    return sorted(int(store) for store in stores)


def resolve_stores(selection: str, data_dir: Path = Path("data/raw")) -> list[int]:
    """`sample` is the pinned 12-store benchmark, `all` is every store in the
    data, and a comma-separated list is taken literally."""
    if selection == "sample":
        return list(SAMPLE_STORES)
    if selection == "all":
        return all_store_ids(data_dir)
    return [int(part) for part in selection.split(",") if part.strip()]
