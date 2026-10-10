from pathlib import Path
import pandas as pd

ROOT = Path("data/flickr8k")
TEXT_ROOT = ROOT / "text"
IMAGE_ROOT = ROOT / "Images"

TOKEN_FILE = TEXT_ROOT / "Flickr8k.token.txt"
TRAIN_FILE = TEXT_ROOT / "Flickr_8k.trainImages.txt"
VAL_FILE = TEXT_ROOT / "Flickr_8k.devImages.txt"
TEST_FILE = TEXT_ROOT / "Flickr_8k.testImages.txt"

OUTPUT = ROOT / "captions_official.csv"


def read_image_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


train_set = read_image_list(TRAIN_FILE)
val_set = read_image_list(VAL_FILE)
test_set = read_image_list(TEST_FILE)

rows = []

with open(TOKEN_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.rstrip("\n")

        if not line or "\t" not in line:
            continue

        image_token, caption = line.split("\t", 1)
        image_name = image_token.split("#")[0]

        if image_name in train_set:
            split = "train"
        elif image_name in val_set:
            split = "val"
        elif image_name in test_set:
            split = "test"
        else:
            continue

        rows.append({
            "image": image_name,
            "caption": caption.strip(),
            "split": split
        })


df = pd.DataFrame(rows)

# 检查不存在的图片
exists_mask = df["image"].map(
    lambda x: (IMAGE_ROOT / x).is_file()
)

missing = sorted(
    df.loc[~exists_mask, "image"].unique()
)

print("Missing official images:", len(missing))

for x in missing[:20]:
    print("  ", x)

# 删除不存在文件对应的 caption
df = df.loc[exists_mask].copy()

# 删除重复 caption
# df = df.drop_duplicates(
#     subset=["image", "caption", "split"]
# ).reset_index(drop=True)
df = df.reset_index(drop=True)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("\nImages:")
print(df.groupby("split")["image"].nunique())

print("\nCaption pairs:")
print(df.groupby("split").size())

print("\nTotal unique images:", df["image"].nunique())
print("Total caption pairs:", len(df))
print("\nSaved:", OUTPUT)