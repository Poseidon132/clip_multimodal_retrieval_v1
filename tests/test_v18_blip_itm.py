"""Offline contract tests: mining, leakage, checkpoint safety, optimizer smoke."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.blip_itm_training import (
    load_blip_itm_adapter, mine_training_pairs, save_blip_itm_adapter,
    select_trainable_blip_parameters,
)
from src.index_metadata import get_index_metadata
from scripts.train_blip_itm import train


def fake_gallery():
    names = np.array(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    captions = np.array([
        "red bicycle moving fast", "a person rides bicycle",  # a
        "a green forest trail", "green trees and bushes",      # b
        "snow covered mountain peak", "white mountain below clouds",  # c
        "blue lake and evening sky", "water with blue evening sky",  # d
    ])
    targets = np.repeat(np.arange(4), 2)
    positives = [[2*i,2*i+1] for i in range(4)]
    # vectors in a shared 4D space, with some deliberately confusing neighbors
    feat_i = np.eye(4, dtype="float32")
    feat_t = np.array([[1, 0, 0, 0], [.8, .2, 0, 0],
                       [0, 1, 0, 0], [.1, .8, 0, 0],
                       [0, 0, 1, 0], [0, .1, .8, 0],
                       [0, 0, 0, 1], [.1, 0, 0, .9]], dtype="float32")
    return dict(image_names=names, captions=captions, targets=targets,
                positive_texts=positives, image_features=feat_i,
                text_features=feat_t)


def test_mining_balanced_and_no_same_image_false_negative():
    g = fake_gallery()
    pairs, s = mine_training_pairs(g, 4, seed=42, epoch=1, hard_negative_fraction=1.0,
                                   candidate_pool=4, device="cpu")
    assert s["anchor_images"] == 4
    assert s["pairs"] == 8
    assert sum(p["label"] for p in pairs) == 4
    for p in pairs:
        assert p["image"] in g["image_names"], "All negatives are from supplied train gallery"
        assert p["kind"] != "positive" or p["label"] == 1
        if p["kind"] in ("hard_image", "random_image"):
            assert p["image"] != p["anchor_image"]
        if p["kind"] in ("hard_text", "random_text"):
            assert p["image"] == p["anchor_image"]
            assert p["caption"] not in [g["captions"][i] for i in
                                         g["positive_texts"][list(g["image_names"]).index(p["image"])]]


def test_mining_deterministic_and_rejects_invalid_args():
    g = fake_gallery()
    a, _ = mine_training_pairs(g, 3, 15, 1, candidate_pool=3, device="cpu")
    b, _ = mine_training_pairs(g, 3, 15, 1, candidate_pool=3, device="cpu")
    assert a == b
    with pytest.raises(ValueError):
        mine_training_pairs(g, 0, 15, 1)
    with pytest.raises(ValueError):
        mine_training_pairs(g, 3, 15, 1, hard_negative_fraction=1.5)


class FakeTextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
    def forward(self, x):
        return torch.tanh(self.linear(x))


class FakeBlip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(4, 4)
        self.text_encoder = nn.Module()
        self.text_encoder.encoder = nn.Module()
        self.text_encoder.encoder.layer = nn.ModuleList([FakeTextLayer(), FakeTextLayer()])
        self.itm_head = nn.Linear(4, 2)

    def forward(self, pixel_values, input_ids, attention_mask=None, use_itm_head=True):
        p = pixel_values.view(pixel_values.shape[0], 4)
        a = self.vision_model(p)
        b = input_ids.float().view(input_ids.shape[0], 4)
        for layer in self.text_encoder.encoder.layer:
            b = layer(b)
        return types.SimpleNamespace(itm_score=self.itm_head(a + b))


class FakeProcessor:
    def __call__(self, images, text, padding=True, truncation=True, max_length=35, return_tensors="pt"):
        inp = []
        for image in images:
            ar = np.asarray(image.resize((2,2)).convert("L"),dtype=np.float32) / 255.0
            inp.append(ar)
        nums = [[(ord(c) % 17)/17 for c in (s + "xxxx")[:4]] for s in text]
        return dict(pixel_values=torch.tensor(np.stack(inp)), input_ids=torch.tensor(nums))


def test_freezing_and_checkpoint_is_partial_and_safe(tmp_path):
    model = FakeBlip()
    selected = select_trainable_blip_parameters(model, 1)
    assert any(s.startswith("itm_head.") for s in selected)
    assert any(s.startswith("text_encoder.encoder.layer.1.") for s in selected)
    assert not any(s.startswith("vision_model.") for s in selected)
    saved = {k:v.clone() for k,v in model.state_dict().items()}
    path = tmp_path / "blip_adapter.pt"
    save_blip_itm_adapter(model, path, "test/blip", {"learning_rate":5e-6}, 2, .8)
    for n,p in model.named_parameters():
        if p.requires_grad:
            p.data.zero_()
    meta = load_blip_itm_adapter(model, path, "test/blip")
    assert meta == {"epoch":2,"validation_mean_R@1":0.8}
    assert all(torch.equal(model.state_dict()[key],saved[key]) for key in selected)
    with pytest.raises(ValueError,match="base model"):
        load_blip_itm_adapter(model,path,"different/blip")
    assert select_trainable_blip_parameters(model,0) == [k for k,_ in model.itm_head.named_parameters(prefix="itm_head")]
    with pytest.raises(ValueError):
        select_trainable_blip_parameters(model, 99)


def _make_small_project(tmp_path):
    from PIL import Image
    names = ["a.jpg","b.jpg","c.jpg","d.jpg","e.jpg","f.jpg"]
    root = tmp_path / "images"
    root.mkdir()
    for i,name in enumerate(names):
        Image.new("RGB",(4,4),(30*i,20*i,10*i)).save(root/name)
    rows=[]
    for i,name in enumerate(names):
        split = "train" if i < 4 else ("val" if i == 4 else "test")
        for n in range(2):
            rows.append(dict(image=name, caption=f"caption about {name} number {n}",split=split))
    csv=tmp_path / "captions.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    clip_adapter=tmp_path/"clip_adapter.pt"
    clip_adapter.write_bytes(b"dummy-clip-adapter-sha")
    info=get_index_metadata("dummy-clip",""+str(clip_adapter),"image-only")
    index=tmp_path/"index.npz"
    texts=[r["caption"] for r in rows]
    np.savez(index,
        image_names=np.array(names), image_features=np.eye(6,dtype="float32"),
        captions=np.array(texts), caption_images=np.array([r["image"] for r in rows]),
        text_features=np.repeat(np.eye(6,dtype="float32"),2,axis=0),
        **info)
    conf=dict(blip_model="test/blip",clip_model="dummy-clip",clip_index=str(index),
              clip_adapter=str(clip_adapter),data_csv=str(csv),image_root=str(root),
              output_dir=str(tmp_path/"output"),device="cpu",amp="fp32",
              train_last_text_layers=1,anchors_per_epoch=4,batch_size=2,epochs=1,
              hard_negative_fraction=1.0,candidate_pool=6,similar_caption_threshold=1.0)
    import yaml
    cfg=tmp_path/"test.yaml"
    cfg.write_text(yaml.safe_dump(conf),encoding="utf-8")
    return str(cfg)


def test_check_only_and_smoke_with_fake_blip(tmp_path,monkeypatch):
    config_file=_make_small_project(tmp_path)
    check=train(config_file,check_only=True)
    assert check["split_counts"] == {"train":4,"val":1,"test":1}
    fake=types.ModuleType("transformers")
    fake.AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**kw:FakeProcessor())
    fake.BlipForImageTextRetrieval=types.SimpleNamespace(from_pretrained=lambda *a,**kw:FakeBlip())
    monkeypatch.setitem(sys.modules,"transformers",fake)
    history=train(config_file,smoke=True)
    assert len(history)==1 and history[0]["pairs_seen"]>0
    assert not (tmp_path/"output"/"adapter.pt").exists()
    assert (tmp_path/"output"/"smoke"/"train_history.json").is_file()


def test_existing_reranker_can_load_finetuned_checkpoint(tmp_path, monkeypatch):
    from PIL import Image
    from src.blip_rerank import BlipITMScorer
    source = FakeBlip()
    select_trainable_blip_parameters(source, 1)
    save_blip_itm_adapter(source, tmp_path / "adapter.pt", "test/blip", {}, epoch=3, val_mean_r1=.88)
    fake = types.ModuleType("transformers")
    fake.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: FakeProcessor())
    fake.BlipForImageTextRetrieval = types.SimpleNamespace(from_pretrained=lambda *a, **kw: FakeBlip())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    scorer = BlipITMScorer("test/blip", device="cpu", precision="fp32", checkpoint=str(tmp_path / "adapter.pt"))
    for k,v in source.state_dict().items():
        if k.startswith("itm_head.") or k.startswith("text_encoder.encoder.layer.1."):
            assert torch.equal(v, scorer.model.state_dict()[k])
    img = Image.new("RGB", (5, 5), (2, 3, 4))
    scores = scorer.score_pairs([img, img], ["first", "second"], batch_size=1)
    assert scores.shape == (2,) and np.isfinite(scores).all()


def test_val_comparison_rejects_test_and_mismatched_queries(tmp_path):
    import json
    from scripts.compare_blip_val import compare
    base = {
      "split":"val","gallery_images":3,"gallery_captions":6,
      "evaluated_text_queries":6,"evaluated_image_queries":3,
      "candidate_top_k":10,"clip_index":"same.npz","adapter_mode":"image-only",
      "query_seed":2026,"selected":{"alpha":1.0,**{k:0.5 for k in
            ("t2i_R@1","t2i_R@5","t2i_R@10","i2t_R@1","i2t_R@5","i2t_R@10","mean_R@1")}},
      "timing":{}
    }
    a,b=tmp_path/"old.json",tmp_path/"new.json"
    a.write_text(json.dumps(base))
    newer=json.loads(json.dumps(base));newer["selected"]["mean_R@1"]=0.6
    b.write_text(json.dumps(newer))
    assert compare(a,b)["mean_R@1"] == 10.0
    newer["split"]="test";b.write_text(json.dumps(newer))
    with pytest.raises(ValueError):compare(a,b)
    newer["split"]="val";newer["evaluated_text_queries"]=2;b.write_text(json.dumps(newer))
    with pytest.raises(ValueError):compare(a,b)
