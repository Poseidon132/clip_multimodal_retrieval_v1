from __future__ import annotations

import numpy as np
import pytest

from src.blip_rerank import minmax_scores, rerank_top_k, top_k_indices
from scripts.eval_blip_rerank import (
    build_candidates, compute_metrics_for_alpha, sampled_query_indices, subset_gallery,
)


def test_sort_deterministic_and_confidence_alpha():
    assert top_k_indices(np.array([.2, .9, .9, .1]), 3).tolist() == [1, 2, 0]
    ids, values = rerank_top_k([4, 5, 6], [.8, .7, .3], [.02, .99, .1], 1.)
    assert ids.tolist() == [5, 6, 4]
    ids, _ = rerank_top_k([4, 5, 6], [.8, .7, .3], [.02, .99, .1], 0.)
    assert ids.tolist() == [4, 5, 6]
    assert minmax_scores(np.array([3, 3])).tolist() == [0.5, 0.5]


def test_sample_query_indices_are_seeded():
    assert sampled_query_indices(100, 20, 42).tolist() == sampled_query_indices(100, 20, 42).tolist()
    assert len(sampled_query_indices(100, 0, 42)) == 100
    with pytest.raises(ValueError):
        sampled_query_indices(30, -2, 42)


def test_reordering_increases_r1_but_not_r10():
    # 10 candidates; the correct entry starts at rank 2 for both directions.
    similarity = np.eye(10) * 0.9
    similarity[0, 0] = 0.6
    similarity[0, 1] = 0.95
    similarity[1, 0] = 0.97
    tq = np.array([0])
    iq = np.array([0])
    t2i, i2t = build_candidates(similarity, tq, iq, top_k=10)
    assert t2i[0][0][0] == 1
    assert i2t[0][0][0] == 1
    blip_t = [np.array([0.01 if i != 0 else 0.99 for i in t2i[0][0]])]
    blip_i = [np.array([0.01 if i != 0 else 0.99 for i in i2t[0][0]])]
    targets = np.arange(10)
    positives = [[i] for i in range(10)]
    baseline = compute_metrics_for_alpha(t2i, i2t, blip_t, blip_i, 0., tq, iq, targets, positives)
    improved = compute_metrics_for_alpha(t2i, i2t, blip_t, blip_i, 1., tq, iq, targets, positives)
    assert baseline['mean_R@1'] == 0
    assert improved['mean_R@1'] == 1
    assert improved['t2i_R@10'] == baseline['t2i_R@10'] == 1
    assert improved['i2t_R@10'] == baseline['i2t_R@10'] == 1


def test_rerank_bad_probability_rejected():
    with pytest.raises(ValueError, match='probabilities'):
        rerank_top_k([0, 1], [.1, .2], [0., 5.], .5)


def test_subset_gallery_keeps_full_split_and_rejects_mismatch(tmp_path):
    import pandas as pd

    df = pd.DataFrame([
        ('a.jpg', 'a-1', 'val'), ('a.jpg', 'a-2', 'val'),
        ('b.jpg', 'b-1', 'val'), ('b.jpg', 'b-2', 'val'),
        ('c.jpg', 'c-1', 'train'),
    ], columns=['image', 'caption', 'split'])
    csv = tmp_path / 'captions.csv'
    df.to_csv(csv, index=False)
    class Index:
        image_names = np.array(['a.jpg', 'b.jpg', 'c.jpg'])
        image_features = np.eye(3, dtype='float32')
        caption_images = df.image.to_numpy()
        captions = df.caption.to_numpy()
        text_features = np.eye(5, 3, dtype='float32')
    subset = subset_gallery(Index(), str(csv), 'val')
    assert len(subset['image_names']) == 2
    assert len(subset['captions']) == 4
    assert subset['targets'].tolist() == [0, 0, 1, 1]


def test_blip_score_pairs_extracts_positive_class_without_network():
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from src.blip_rerank import BlipITMScorer

    class DummyProcessor:
        def __call__(self, images, text, **kwargs):
            return {'pixel_values': torch.zeros(len(text), 3, 4, 4),
                    'input_ids': torch.ones(len(text), 3, dtype=torch.long)}
    class DummyModel:
        def __call__(self, **kwargs):
            n = kwargs['input_ids'].shape[0]
            return SimpleNamespace(itm_score=torch.tensor([[0., 2.]] * n))
    scorer = BlipITMScorer.__new__(BlipITMScorer)
    scorer.device = torch.device('cpu')
    scorer.dtype = torch.float32
    scorer.processor = DummyProcessor()
    scorer.model = DummyModel()
    im = Image.new('RGB', (4, 4))
    scores = scorer.score_pairs([im] * 3, ['caption'] * 3, batch_size=2)
    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, 1 / (1 + np.exp(-2)), atol=1e-6)


def test_full_small_evaluation_pipeline_with_fake_blip(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from scripts.eval_blip_rerank import run
    from src.index_metadata import get_index_metadata

    names = [f'{i}.jpg' for i in range(10)]
    captions = [f'{i}.jpg caption {j}' for i in range(10) for j in range(2)]
    owners = [n for n in names for _ in range(2)]
    df = pd.DataFrame({'image':owners,'caption':captions,'split':['val']*20})
    csv = tmp_path/'captions.csv'; df.to_csv(csv,index=False)
    adapter = tmp_path/'adapter.pt'; adapter.write_bytes(b'weights')
    metadata = get_index_metadata('fake-clip',str(adapter),'image-only')
    rng = np.random.default_rng(9)
    image_features = rng.normal(size=(10,8)).astype('float32')
    text_features = rng.normal(size=(20,8)).astype('float32')
    index = tmp_path/'index.npz'
    np.savez_compressed(index,image_names=np.array(names),image_features=image_features,
                        captions=np.array(captions),caption_images=np.array(owners),
                        text_features=text_features,**metadata)
    class FakeBLIP:
        def score_pairs(self, images, texts, batch_size):
            return np.array([0.99 if str(img).replace('\\','/').split('/')[-1] == text.split()[0] else 0.01
                             for img,text in zip(images,texts)],dtype='float32')
    args = SimpleNamespace(top_k=10,batch_size=3,split='val',alpha=None,alphas=[0.,1.],
                           sample_text_queries=5,sample_image_queries=5,
                           seed=42,index=str(index),clip_model='fake-clip',adapter=str(adapter),
                           adapter_mode='image-only',csv=str(csv),image_root=str(tmp_path),
                           blip_model='fake-blip',output=str(tmp_path/'output.json'))
    output = run(args,scorer=FakeBLIP())
    assert output['evaluated_text_queries'] == 5
    assert output['evaluated_image_queries'] == 5
    assert output['selected']['t2i_R@1'] == 1.
    assert output['selected']['i2t_R@1'] == 1.
    assert output['selected']['alpha'] == 1.
    assert (tmp_path/'output.json').is_file()
