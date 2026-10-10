from __future__ import annotations

import numpy as np


def recall_at_k_text_to_image(similarity: np.ndarray, target_image_indices: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    order = np.argsort(-similarity, axis=1)
    out = {}
    for k in ks:
        hits = [target_image_indices[i] in order[i, :k] for i in range(similarity.shape[0])]
        out[f"t2i_R@{k}"] = float(np.mean(hits))
    return out


def recall_at_k_image_to_text(similarity: np.ndarray, positive_text_indices: list[list[int]], ks=(1, 5, 10)) -> dict[str, float]:
    order = np.argsort(-similarity.T, axis=1)
    out = {}
    for k in ks:
        hits = []
        for image_idx, positives in enumerate(positive_text_indices):
            topk = set(order[image_idx, :k].tolist())
            hits.append(bool(topk.intersection(positives)))
        out[f"i2t_R@{k}"] = float(np.mean(hits))
    return out


def retrieval_metrics(similarity: np.ndarray, target_image_indices: np.ndarray, positive_text_indices: list[list[int]], ks=(1, 5, 10)) -> dict[str, float]:
    result = {}
    result.update(recall_at_k_text_to_image(similarity, target_image_indices, ks))
    result.update(recall_at_k_image_to_text(similarity, positive_text_indices, ks))
    result["mean_R@1"] = 0.5 * (result["t2i_R@1"] + result["i2t_R@1"])
    return result
