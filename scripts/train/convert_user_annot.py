# -*- coding: utf-8 -*-
"""用户标注 -> 训练数据集（小地图 + 全屏）。

用户标注文件：temp/ann/s*.txt（YOLO 格式，类别 0-10 全屏 / 11-19 小地图）
输出：
  data/mm_user_v7/    小地图训练集（12 张 + 小地图归一化 txt，类别 0-8）
  data/yolo_dataset/zhongkui_user/  全屏训练集（12 张 + 全屏 txt，并入现有数据）
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # scripts/train -> 仓库根
sys.path.insert(0, str(ROOT))

MM_W, MM_H = 232, 232          # 小地图尺寸（全屏 1280x720 标定）
FULL_W, FULL_H = 1280, 720

# 小地图类别映射（标注类 11-19 -> 训练类 0-8）
MM_MAP = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8}
MM_NAMES = ["mm_self", "mm_ally", "mm_enemy", "mm_ally_tower", "mm_enemy_tower",
            "mm_monster", "mm_buff", "mm_ally_minion", "mm_enemy_minion"]


def convert(src_dir, mm_out, full_out):
    src_dir = ROOT / src_dir
    for out in (mm_out, full_out):
        for split in ("train", "val"):
            for sub in ("images", "labels"):
                d = ROOT / out / sub / split
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

    mm_labels = {}
    full_labels = {}
    for txt in sorted(src_dir.glob("*.txt")):
        img_p = txt.with_suffix(".png")
        if not img_p.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        mm_lines = []
        full_lines = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if c <= 10:
                full_lines.append(f"{c} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
            elif c in MM_MAP:
                # 全屏归一化 -> 小地图像素 -> 小地图归一化
                px = cx * w
                py = cy * h
                mm_lines.append(
                    f"{MM_MAP[c]} {px / MM_W:.4f} {py / MM_H:.4f} "
                    f"{bw * w / MM_W:.4f} {bh * h / MM_H:.4f}")
        if mm_lines:
            mm_labels[img_p] = mm_lines
        if full_lines:
            full_labels[img_p] = full_lines

    # 小地图：全部 train（12 张太少，val 复用增强）
    for img_p, lines in mm_labels.items():
        shutil.copy(img_p, ROOT / mm_out / "images" / "train" / img_p.name)
        (ROOT / mm_out / "labels" / "train" / (img_p.stem + ".txt")).write_text(
            "\n".join(lines), encoding="utf-8")
    # 全屏：train/val 划分
    items = list(full_labels.items())
    n_train = max(1, int(len(items) * 0.8))
    for i, (img_p, lines) in enumerate(items):
        split = "train" if i < n_train else "val"
        shutil.copy(img_p, ROOT / full_out / "images" / split / img_p.name)
        (ROOT / full_out / "labels" / split / (img_p.stem + ".txt")).write_text(
            "\n".join(lines), encoding="utf-8")

    # data.yaml
    mm_yaml = f"path: {(ROOT / mm_out).resolve()}\ntrain: images/train\nval: images/train\nnames:\n"
    for i, n in enumerate(MM_NAMES):
        mm_yaml += f"  {i}: {n}\n"
    (ROOT / mm_out / "data.yaml").write_text(mm_yaml, encoding="utf-8")

    print(f"小地图: {len(mm_labels)} 张 -> {mm_out}")
    print(f"全屏: {len(full_labels)} 张 -> {full_out}")


if __name__ == "__main__":
    convert("temp/ann", "data/mm_user_v7", "data/yolo_dataset/zhongkui_user")
