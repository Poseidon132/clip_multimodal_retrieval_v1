import json

import pytest

from scripts.report_experiments import load_metric_result, write_results


def build_metrics():
    return {f'{prefix}_R@{k}':0.5 for prefix in ('t2i','i2t') for k in (1,5,10)} | {'mean_R@1':.5}


def test_reports_and_rejects_wrong_split_or_subsample(tmp_path):
    data = {'split':'val','num_images':1000,'num_captions':5000,**build_metrics()}
    path = tmp_path/'full.json'
    path.write_text(json.dumps(data))
    result = load_metric_result('CLIP',path,'val')
    assert result['mean_R@1'] == 50
    write_results([result], tmp_path/'out')
    assert (tmp_path/'out'/'recall_at_1.png').is_file()
    assert (tmp_path/'out'/'retrieval_comparison.csv').is_file()
    with pytest.raises(ValueError,match='split'):
        load_metric_result('CLIP',path,'test')
    partial = {'split':'val','gallery_images':1000,'gallery_captions':5000,
               'evaluated_text_queries':100,'evaluated_image_queries':20,
               'selected': {'alpha':.5,**build_metrics()},'results':[]}
    path.write_text(json.dumps(partial))
    with pytest.raises(ValueError,match='partial'):
        load_metric_result('BLIP sample',path,'val')
