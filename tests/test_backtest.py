from pathlib import Path

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


# Store selection ---------------------------------------------------------------


def test_sample_selection_is_the_pinned_benchmark():
    from foresight.config import SAMPLE_STORES, resolve_stores

    assert resolve_stores("sample") == SAMPLE_STORES


def test_a_literal_list_is_taken_as_given():
    from foresight.config import resolve_stores

    assert resolve_stores("85,262, 3") == [85, 262, 3]


def test_all_selection_reads_every_store_in_the_committed_sample():
    """Uses the sample file, because the full dataset is gitignored and this has
    to pass on a clean checkout."""
    from foresight.config import all_store_ids

    stores = all_store_ids(Path("data/raw"), train_filename="sample_train.csv")
    assert stores == [1, 2, 3, 85, 262]


def test_all_selection_covers_every_store_when_the_full_dataset_is_present():
    import pytest

    from foresight.config import resolve_stores

    if not Path("data/raw/train.csv").exists():
        pytest.skip("full Rossmann dataset not present")

    stores = resolve_stores("all")
    assert len(stores) == 1115
    assert stores[0] == 1 and stores[-1] == 1115


def test_store_ids_are_not_logged_as_a_five_thousand_character_parameter():
    """MLflow caps parameter length, so the full list for a 1,115-store run has to
    be summarised or the logging call fails partway through a finished run."""
    from foresight.tracking import _store_ids_param

    short = _store_ids_param([1, 2, 3])
    long = _store_ids_param(list(range(1, 1116)))

    assert short == "1,2,3"
    assert len(long) < 100
    assert "1115" in long


def test_the_writeup_states_the_store_count_it_actually_measured():
    """A full run and a sample run must not end up with the same claim in their
    generated writeups."""
    from foresight.backtest import _scope_sentence

    def results(n):
        return {"folds": [{"models": {"lightgbm": {"n_stores": n}, "prophet": {"n_stores": n}}}]}

    assert "12-store benchmark" in _scope_sentence(results(12))
    assert "1,115 stores" in _scope_sentence(results(1115))


def test_a_mismatch_in_store_counts_is_flagged_rather_than_averaged_quietly():
    from foresight.backtest import _scope_sentence

    mixed = {"folds": [{"models": {"lightgbm": {"n_stores": 1115}, "prophet": {"n_stores": 900}}}]}
    assert "did not all score the same" in _scope_sentence(mixed)
