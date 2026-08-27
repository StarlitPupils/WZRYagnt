# -*- coding: utf-8 -*-
"""构建扩充数据集 zhongkui_v3: 伪标注45帧(血条英雄+模型) + 用户真值12帧。

划分: train = 伪45 + s04-s12; val = s01-s03 (用户真值, 独立评估)
"""
import sys
from pathlib import Path
import shutil
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PSEUDO = ROOT / "temp" / "pseudo1"
GT = ROOT / "temp" / "ann"
OLD = ROOT / "data" / "yolo_dataset" / "zhongkui"
DST = ROOT / "data" / "yolo_dataset" / "zhongkui_v3"
VAL_IDS = {1, 2, 3}   # s01-s03 作 val

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (DST / sub).mkdir(parents=True, exist_ok=True)

def copy_img_label(img_path, label_text, split):
    name = img_path.stem
    shutil.copy2(str(img_path), str(DST / "images" / split / f"{name}.png"))
    (DST / "labels" / split / f"{name}.txt").write_text(label_text, encoding="utf-8")

# 0) 原始 448 帧(旧数据集) → train (前缀 o_ 防重名)
n_old = 0
for lp in sorted((OLD / "labels" / "train").glob("*.txt")):
    img_path = OLD / "images" / "train" / (lp.stem + ".jpg")
    if not img_path.exists():
        img_path = OLD / "images" / "train" / (lp.stem + ".png")
    if not img_path.exists():
        continue
    lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if len(ln.split()) == 5]
    if not lines:
        continue
    name = "o_" + lp.stem
    shutil.copy2(str(img_path), str(DST / "images" / "train" / f"{name}.png"))
    (DST / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    n_old += 1

# 1) 伪标注帧 → train
n_pseudo = 0
for lp in sorted(PSEUDO.glob("labels/*.txt")):
    img_path = PSEUDO / "images" / (lp.stem + ".png")
    if not img_path.exists():
        continue
    lines = []
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) == 5:
            lines.append(ln)
    if not lines:
        continue
    copy_img_label(img_path, "\n".join(lines), "train")
    n_pseudo += 1

# 2) 用户真值帧: s01-s03 → val, s04-s12 → train
n_gt = 0
for i in range(1, 13):
    img_path = GT / f"s{i:02d}.png"
    txt_path = GT / f"s{i:02d}.txt"
    if not img_path.exists() or not txt_path.exists():
        continue
    lines = []
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) == 5 and int(p[0]) <= 10:
            lines.append(ln)
    split = "val" if i in VAL_IDS else "train"
    copy_img_label(img_path, "\n".join(lines), split)
    n_gt += 1

yaml = f"""# zhongkui_v3 扩充数据集 (伪标注45 + 用户真值12)
path: {str(DST).replace(chr(92), '/')}
train: images/train
val: images/val
nc: 11
names: ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion", "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal", "neutral_monster", "hook_aim", "skill_effect"]
"""
(DST / "data.yaml").write_text(yaml, encoding="utf-8")

# 统计
from collections import Counter
cnt = Counter()
for split in ("train", "val"):
    for lp in (DST / "labels" / split).glob("*.txt"):
        for ln in lp.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) == 5:
                cnt[(split, int(p[0]))] += 1
print(f"原始帧={n_old} 伪标注帧={n_pseudo} 真值帧={n_gt}")
for split in ("train", "val"):
    print(f"[{split}] " + " ".join(f"c{c}:{cnt.get((split, c), 0)}" for c in range(11)))
