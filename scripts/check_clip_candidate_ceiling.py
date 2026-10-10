"""Fast validation-only CLIP candidate coverage check (no BLIP model loading).

Ranks the full validation gallery once and reports the ideal R@1 upper bound
for reranking Top-10, Top-20 and Top-30 candidates. This is NOT measured BLIP
performance: a real reranker can score lower than these upper bounds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_blip_rerank import build_candidates, candidate_oracle_hit_rates
from src.search import ExactMultimodalIndex
from scripts.eval_blip_rerank import subset_gallery


def evaluate_candidate_ceilings(gallery: dict, cutoffs=(10, 20, 30)) -> dict:
    """Check all queries against the full gallery; no test split or sampling."""
    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or any(k < 1 for k in ks):
        raise ValueError('Candidate cutoffs must be positive integers')
    tq = np.arange(len(gallery['captions']), dtype=np.int64)
    iq = np.arange(len(gallery['image_names']), dtype=np.int64)
    scores = gallery['text_features'] @ gallery['image_features'].T
    t_all, i_all = build_candidates(scores, tq, iq, max(ks))
    rows = []
    for k in ks:
        t = [(ids[:k], cosine[:k]) for ids, cosine in t_all]
        i = [(ids[:k], cosine[:k]) for ids, cosine in i_all]
        oracle = candidate_oracle_hit_rates(
            t, i, tq, iq, gallery['targets'], gallery['positive_texts']
        )
        rows.append({'top_k': k, **oracle})
    return {'gallery_images': len(iq), 'gallery_captions': len(tq), 'cutoffs': rows}


def main():
    p = argparse.ArgumentParser(description='CLIP candidate oracle ceiling for validation only')
    p.add_argument('--index', default='artifacts/flickr8k_visual_multi_index.npz')
    p.add_argument('--csv', default='data/flickr8k/captions_official.csv')
    p.add_argument('--adapter', default='artifacts/flickr8k_visual_multi/adapter.pt')
    p.add_argument('--adapter-mode', default='image-only', choices=['image-only', 'both'])
    p.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    p.add_argument('--cutoffs', nargs='+', type=int, default=[10, 20, 30])
    a = p.parse_args()
    idx = ExactMultimodalIndex(a.index)
    idx.validate_encoder(a.clip_model, a.adapter, a.adapter_mode)
    gallery = subset_gallery(idx, a.csv, 'val')
    r = evaluate_candidate_ceilings(gallery, a.cutoffs)
    print(f"VAL gallery: {r['gallery_images']} images, {r['gallery_captions']} captions")
    print('CLIP candidate coverage (theoretical ideal-reranker R@1 ceiling):')
    print('Top-K   Text->Image     Image->Text     Mean')
    for row in r['cutoffs']:
        print(f"{row['top_k']:>5}   {row['t2i_candidate_hit_rate']*100:>11.2f}%"
              f"     {row['i2t_candidate_hit_rate']*100:>11.2f}%"
              f"   {row['mean_candidate_hit_rate']*100:>7.2f}%")
    print('These ceilings are NOT measured BLIP accuracy; a larger Top-K may worsen actual R@1.')


if __name__ == '__main__':
    main()
