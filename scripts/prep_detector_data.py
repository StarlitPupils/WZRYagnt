# -*- coding: utf-8 -*-
"""整理标注素材 -> YOLO 检测数据集 (v4.3).

小地图标注 (in_mm, 类别 12-17) 与全屏标注 (out_mm, 类别 0-11) 统一:
  - 全屏: 0 enemy_hero 1 ally_hero 8 neutral_monster 11 self (用户实际标的)
  - 小地图: 12 mm_red 13 mm_blue 14 mm_green 15 mm_yellow
    (小地图裁剪 240x240, 作为独立检测任务的"小地图点检测"素材)

输出:
  data/yolo_v4/images/{train,val}/*.png   (全屏 1280x720)
  data/yolo_v4/mm_images/{train,val}/*.png (小地图 240x240)
  对应 labels/*.txt
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)

# 只保留有效标注的类别 (用户实际标的)
FULL_MAP = {0: 0, 1: 1, 8: 8, 11: 11}     # 全屏: enemy_hero/ally_hero/neutral_monster/self
MM_MAP = {12: 12, 13: 13, 14: 14, 15: 15}  # 小地图: mm_red/blue/green/yellow

# 9 类合集 (与检测模型训练一致)
CLASSES = {0: "enemy_hero", 1: "ally_hero", 8: "neutral_monster", 11: "self",
           12: "mm_red", 13: "mm_blue", 14: "mm_green", 15: "mm_yellow"}

OUT_FULL = ROOT / "data" / "yolo_v4"
OUT_MM = ROOT / "data" / "yolo_v4" / "mm"


def collect():
    full = []
    d = ROOT / "data" / "labeling" / "out_mm" / "images"
    for imgf in sorted(d.glob("*.png")):
        txt = imgf.with_suffix(".txt")
        if not txt.exists():
            continue
        lines = []
        for ln in txt.read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            if c in FULL_MAP:
                lines.append(f"{FULL_MAP[c]} {' '.join(p[1:])}")
        if lines:
            full.append((imgf, lines))
    mm = []
    d2 = ROOT / "data" / "labeling" / "in_mm"
    for imgf in sorted(d2.glob("*.png")):
        txt = imgf.with_suffix(".txt")
        if not txt.exists():
            continue
        lines = []
        for ln in txt.read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            if c in MM_MAP:
                lines.append(f"{MM_MAP[c]} {' '.join(p[1:])}")
        if lines:
            mm.append((imgf, lines))
    return full, mm


def write_dataset(items, out_dir, name):
    train_d = out_dir / "images" / "train"
    val_d = out_dir / "images" / "val"
    train_l = out_dir / "labels" / "train"
    val_l = out_dir / "labels" / "val"
    for d in (train_d, val_d, train_l, val_l):
        d.mkdir(parents=True, exist_ok=True)
    # 清理旧
    for d in (train_d, val_d, train_l, val_l):
        for f in d.glob("*"):
            f.unlink()
    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.2))
    val_items = items[:n_val]
    train_items = items[n_val:]
    for imgf, lines in train_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        cv2.imwrite(str(train_d / imgf.name), img)
        (train_l / f"{imgf.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    for imgf, lines in val_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        cv2.imwrite(str(val_d / imgf.name), img)
        (val_l / f"{imgf.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{name}: train {len(train_items)} val {len(val_items)}")


if __name__ == "__main__":
    full, mm = collect()
    print(f"全屏标注: {len(full)} 张, 小地图标注: {len(mm)} 张")
    write_dataset(full, OUT_FULL, "全屏")
    write_dataset(mm, OUT_MM, "小地图")
    print("数据集就绪: data/yolo_v4/ (全屏) + data/yolo_v4/mm/ (小地图)")
