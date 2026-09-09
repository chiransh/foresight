import pandas as pd

from foresight.backtest import make_folds


def test_folds_are_contiguous_and_non_overlapping():
    max_date = pd.Timestamp("2015-07-31")
    folds = make_folds(max_date, n_folds=3, fold_size=42)

    assert len(folds) == 3
    # Chronological order: fold 1 is earliest.
    for earlier, later in zip(folds, folds[1:]):
        assert earlier["test_end"] + pd.Timedelta(days=1) == later["test_start"]


def test_training_window_expands_across_folds():
    max_date = pd.Timestamp("2015-07-31")
    folds = make_folds(max_date, n_folds=3, fold_size=42)

    train_ends = [fold["train_end"] for fold in folds]
    assert train_ends == sorted(train_ends)  # strictly increasing, each fold sees more history


def test_last_fold_matches_the_single_holdout_convention():
    max_date = pd.Timestamp("2015-07-31")
    folds = make_folds(max_date, n_folds=3, fold_size=42)

    last_fold = folds[-1]
    assert last_fold["test_end"] == max_date
    assert last_fold["train_end"] == max_date - pd.Timedelta(days=42)
