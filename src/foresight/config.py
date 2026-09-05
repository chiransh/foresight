"""Shared config for the baseline models so their metrics are comparable.

SAMPLE_STORES is a fixed set of 12 stores stratified across all four
StoreType values and all three Assortment levels (random_state=42 over
store.csv, 3 per type). Fitting all three model families against every one
of the 1,115 stores isn't necessary to compare them honestly, but fitting
each against a different subset would be: this list is what SARIMA/Prophet,
LightGBM, and NHITS/TFT all train and evaluate against, so Day 7's
head-to-head table is comparing the same stores, not different samples.
"""

SAMPLE_STORES = [85, 195, 259, 347, 353, 464, 772, 791, 982, 1006, 1053, 1070]

HOLDOUT_DAYS = 42
