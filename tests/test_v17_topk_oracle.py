"""Regression coverage for expanding reranking beyond the original Top-10."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.eval_blip_rerank import (
    candidate_oracle_hit_rates, compute_metrics_for_alpha, run,
)
from src.index_metadata import get_index_metadata


def test_candidate_oracle_and_r10_can_change_with_top20():
    # Correct item starts outside CLIP's Top-10 but inside Top-20.
    correct = 19
    candidate_ids = np.arange(20)
    clip_scores = np.linspace(1., 0., 20)
    candidates = [(candidate_ids, clip_scores)]
    blip_scores = [np.array([1. if i == correct else 0. for i in candidate_ids])]
    queries = np.array([0])
    targets = np.array([correct])
    positives = [[correct]]

    oracle = candidate_oracle_hit_rates(candidates, candidates, queries, queries, targets, positives)
    assert oracle["t2i_candidate_hit_rate"] == 1.
    assert oracle["i2t_candidate_hit_rate"] == 1.

    baseline = compute_metrics_for_alpha(candidates, candidates, blip_scores, blip_scores,
                                         0., queries, queries, targets, positives)
    reranked = compute_metrics_for_alpha(candidates, candidates, blip_scores, blip_scores,
                                         1., queries, queries, targets, positives)
    assert baseline["t2i_R@10"] == 0.
    assert reranked["t2i_R@10"] == 1.
    assert reranked["mean_R@1"] == 1.


def test_full_run_top20_supports_changed_recall10(tmp_path):
    n = 20
    image_names = [f"{i}.jpg" for i in range(n)]
    owners = [name for name in image_names for _ in range(2)]
    captions = [f"{i}.jpg caption {j}" for i in range(n) for j in range(2)]
    df = pd.DataFrame({"image": owners, "caption": captions, "split": ["val"] * len(owners)})
    csv = tmp_path / "captions.csv"
    df.to_csv(csv, index=False)
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"mock")
    meta = get_index_metadata("mock-clip", str(adapter), "image-only")
    # Flat features: every query starts with early-gallery candidates.
    # Fake BLIP reorders the correct match to the front whenever present.
    features_image = np.ones((n, 4), dtype=np.float32)
    features_text = np.ones((len(captions), 4), dtype=np.float32)
    path = tmp_path / "index.npz"
    np.savez_compressed(path, image_names=np.array(image_names),
                        image_features=features_image,
                        captions=np.array(captions), caption_images=np.array(owners),
                        text_features=features_text, **meta)

    class FakeScorer:
        def score_pairs(self, images, texts, batch_size):
            return np.array([1. if img.stem == text.split()[0].removesuffix('.jpg') else 0.
                             for img, text in zip(images, texts)], dtype=np.float32)

    args = SimpleNamespace(top_k=20, batch_size=4, split="val", alpha=1., alphas=[1.],
                           sample_text_queries=0, sample_image_queries=0, seed=42,
                           index=str(path), clip_model="mock-clip", adapter=str(adapter),
                           adapter_mode="image-only", csv=str(csv), image_root=str(tmp_path),
                           blip_model="fake-itm", output=str(tmp_path / "output.json"))
    result = run(args, scorer=FakeScorer())
    assert result["candidate_top_k"] == 20
    assert result["candidate_oracle_at_top_k"]["t2i_candidate_hit_rate"] == 1.
    assert result["selected"]["t2i_R@10"] > result["clip_baseline"]["t2i_R@10"]
    assert result["selected"]["t2i_R@1"] > result["clip_baseline"]["t2i_R@1"]


def test_fast_candidate_precheck_is_non_decreasing():
    from scripts.check_clip_candidate_ceiling import evaluate_candidate_ceilings

    # Two queries, four images, at least one positive among later candidates.
    image_features = np.eye(4, dtype=np.float32)
    text_features = np.array([[0., 0., .5, 1.], [1., 0., 0., 0.]], dtype=np.float32)
    gallery = {
        'image_names': np.array(['0.jpg','1.jpg','2.jpg','3.jpg']),
        'image_features': image_features,
        'captions': np.array(['q1','q2']),
        'text_features': text_features,
        'targets': np.array([2, 0], dtype=np.int64),
        'positive_texts': [[1], [], [0], []],
    }
    result = evaluate_candidate_ceilings(gallery, (1, 2, 4))
    rows = result['cutoffs']
    assert [r['top_k'] for r in rows] == [1,2,4]
    for field in ('t2i_candidate_hit_rate','i2t_candidate_hit_rate','mean_candidate_hit_rate'):
        assert all(a[field] <= b[field] for a,b in zip(rows, rows[1:]))
