from src.dataset import load_caption_table


def test_toy_table_schema():
    df = load_caption_table("data/toy/captions.csv")
    assert set(["image", "caption", "split"]).issubset(df.columns)
    assert len(df) > 0
