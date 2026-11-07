import os, random, shutil
from pathlib import Path
import os, random, shutil
from pathlib import Path

SRC = Path("./coco128")
DST = Path("./coco_person_car")
KEEP = {0:0, 2:1}
VAL_RATIO = 0.2

img_dir = SRC / "images" / "train2017"
lbl_dir = SRC / "labels" / "train2017"

# 先把所有真实存在的图片名字存下来
exist_img_stems = {p.stem for p in img_dir.glob("*.jpg")}
pool = []

for lbl in sorted(lbl_dir.glob("*.txt")):
    stem = lbl.stem
    # 如果这张图本身不存在，直接跳过这个标签
    if stem not in exist_img_stems:
        print(f"跳过：标签 {lbl.name}，找不到对应的图片")
        continue

    rows = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sp = line.split()
        cid = int(sp[0])
        if cid in KEEP:
            rows.append(f"{KEEP[cid]} {' '.join(sp[1:])}")

    if len(rows) > 0:
        pool.append(stem)
        (lbl.parent / (stem + ".tmp")).write_text("\n".join(rows))

random.seed(0)
random.shuffle(pool)
n_val = int(len(pool) * VAL_RATIO)
val_list = pool[:n_val]
train_list = pool[n_val:]

for split, filelist in zip(["train", "val"], [train_list, val_list]):
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    for name in filelist:
        shutil.copy(img_dir / f"{name}.jpg", DST / "images" / split / f"{name}.jpg")
        shutil.copy(lbl_dir / f"{name}.tmp", DST / "labels" / split / f"{name}.txt")

# 清理临时文件
for f in lbl_dir.glob("*.tmp"):
    f.unlink()

print(f"\n✅处理完成！训练集 {len(train_list)} 张，验证集 {len(val_list)} 张")

