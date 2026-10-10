import numpy as np

from src.metrics import retrieval_metrics


def test_perfect_retrieval():
    similarity = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    target = np.array([0, 0, 1, 1])
    positives = [[0, 1], [2, 3]]
    m = retrieval_metrics(similarity, target, positives, ks=(1, 2))
    assert m["t2i_R@1"] == 1.0
    assert m["i2t_R@1"] == 1.0
