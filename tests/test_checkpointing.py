import pytest

from src.checkpointing import BestScoreTracker


def test_improvement_and_patience():
    t = BestScoreTracker(patience=2, min_delta=0.001)
    assert t.step(0.60, 1) == (True, False)
    assert t.step(0.6005, 2) == (False, False)
    assert t.step(0.6006, 3) == (False, True)
    assert t.best_epoch == 1
    assert t.best_score == 0.60


def test_improvement_resets_patience():
    t = BestScoreTracker(patience=2)
    assert t.step(0.60, 1) == (True, False)
    assert t.step(0.59, 2) == (False, False)
    assert t.step(0.62, 3) == (True, False)
    assert t.step(0.62, 4) == (False, False)
    assert t.best_epoch == 3


def test_patience_zero_disables_early_stopping():
    t = BestScoreTracker(patience=0)
    assert t.step(0.5, 1) == (True, False)
    assert t.step(0.4, 2) == (False, False)


def test_bad_metric_rejected():
    t = BestScoreTracker()
    with pytest.raises(ValueError):
        t.step(float("nan"), 1)
