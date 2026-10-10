from __future__ import annotations

import csv
import io

import pytest

from src.release_utils import read_report_rows, rows_to_csv_bytes, excel_safe_cell, format_score
from scripts.build_release_report import read_artifact, fpercent, merge_source_metrics


def test_bom_csv_missing_metrics(tmp_path):
    f = tmp_path / "metrics.csv"
    f.write_text("\ufeffsplit,model,t2i_R@1,t2i_R@5,i2t_R@1\nval,A,60.32,,75.90\n", encoding="utf-8")
    rows = read_report_rows(f)
    assert rows[0]["t2i_R@1"] == 60.32
    assert rows[0]["t2i_R@5"] is None
    assert rows[0]["i2t_R@1"] == 75.90


def test_excel_formula_export_escaped():
    data = rows_to_csv_bytes([{"caption": "=HYPERLINK(\"x\")", "score": 0.5}], ["caption", "score"])
    assert data.startswith(b"\xef\xbb\xbf")
    decoded = data.decode("utf-8-sig")
    line = next(csv.DictReader(io.StringIO(decoded)))
    assert line["caption"].startswith("'=HYPERLINK")


def test_normal_text_export_no_change():
    assert excel_safe_cell("a person jumping") == "a person jumping"
    assert format_score(.73448, True) == "0.734"
    assert format_score(.73448, False) == "0.7345"


def test_read_full_blip_json_and_source_clip_baseline(tmp_path):
    art = tmp_path / "artifacts" / "blip.json"
    art.parent.mkdir(parents=True)
    import json
    keys = ["t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_R@1"]
    metrics = {k: 0.75 for k in keys}
    art.write_text(json.dumps({"split":"test","gallery_images":1000,"gallery_captions":5000,
        "evaluated_text_queries":5000,"evaluated_image_queries":1000,
        "selected":metrics,"clip_baseline":{k:.5 for k in keys},"results":[metrics]}), encoding="utf-8")
    assert read_artifact("artifacts/blip.json", tmp_path, "test")["t2i_R@1"] == 75.0
    assert read_artifact("artifacts/blip.json#clip_baseline", tmp_path, "test")["t2i_R@1"] == 50.0


def test_partial_json_rejected(tmp_path):
    art = tmp_path / "short.json"
    import json
    art.write_text(json.dumps({"split":"val","gallery_images":1000,"gallery_captions":5000,
      "evaluated_text_queries":100,"evaluated_image_queries":20,"selected":{},"results":[]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Partial evaluation"):
        read_artifact("short.json", tmp_path, "val")


def test_missing_not_invented(tmp_path):
    rows = [{"split":"test","model":"A","source_artifact":"nonexistent.json",
             "t2i_R@1":"79.32", "t2i_R@5":"", "i2t_R@1":"90.70"}]
    assert merge_source_metrics(rows, tmp_path) == []
    assert rows[0]["t2i_R@5"] == ""
    assert fpercent(rows[0], "t2i_R@5") == "—"
