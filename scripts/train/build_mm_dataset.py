# -*- coding: utf-8 -*-
"""小地图元素 YOLO 训练集合成器（用户提供抠图 -> 合成数据 -> YOLO 训练集）。

类别（9 类，用户指定）：
  0 mm_self          自己绿圈
  1 mm_ally          队友蓝圈
  2 mm_enemy         敌人红圈
  3 mm_ally_tower    蓝方塔
  4 mm_enemy_tower   红方塔
  5 mm_monster       野怪点
  6 mm_buff          buff 标记
  7 mm_ally_minion   我方小兵
  8 mm_enemy_minion  敌方小兵

目录结构：
  data/mm_cutouts/<class>/  用户提供的抠图（PNG/JPG，单元素）
  data/mm_backgrounds/      真实小地图背景（自动提取）
  data/mm_dataset/          YOLO 训练集（images/labels + data.yaml）

用法：
  venv\\Scripts\\python.exe scripts\\train\\build_mm_dataset.py [--per-img 8] [--imgs 600]
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CLASSES = ["mm_self", "mm_ally", "mm_enemy",
           "mm_ally_tower", "mm_enemy_tower",
           "mm_monster", "mm_buff",
           "mm_ally_minion", "mm_enemy_minion"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_cutouts(cutout_dir: Path):
    """加载各分类抠图。返回 {cls: [img, ...]}（RGBA 或 BGR）。"""
    pool = {}
    for cls in CLASSES:
        d = cutout_dir / cls
        imgs = []
        if d.exists():
            for f in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                data = np.fromfile(str(f), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    imgs.append(img)
        if imgs:
            pool[cls] = imgs
            print(f"  {cls}: {len(imgs)} 张")
        else:
            print(f"  {cls}: 无（跳过）")
    return pool


def paste(front, back, x, y, size):
    """把抠图贴到背景 (x, y) 处，缩放为 size。支持 alpha 通道。"""
    f = cv2.resize(front, (size, size), interpolation=cv2.INTER_AREA)
    h, w = back.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return
    fh, fw = y1 - y0, x1 - x0
    ff = f[0:fh, 0:fw]
    if ff.ndim == 3 and ff.shape[2] == 4:
        alpha = ff[..., 3:4].astype(np.float32) / 255.0
        rgb = ff[..., :3].astype(np.float32)
        back[y0:y1, x0:x1] = (rgb * alpha
                              + back[y0:y1, x0:x1].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    else:
        back[y0:y1, x0:x1] = ff


def synthesize_one(bg, pool, rng, per_img):
    """合成一张训练图：1 背景 + 若干元素，返回 (img, labels)。"""
    img = bg.copy()
    h, w = img.shape[:2]
    labels = []  # (cls_id, cx, cy, bw, bh) 归一化
    placed = 0
    tries = 0
    while placed < per_img and tries < per_img * 8:
        tries += 1
        cls = rng.choice(list(pool.keys()))
        cut = rng.choice(pool[cls])
        # 尺寸：英雄圈 ~26-34px, 塔 ~10-16px, 小兵 ~6-10px（相对 232px 小地图）
        base = {"mm_self": 30, "mm_ally": 28, "mm_enemy": 28,
                "mm_ally_tower": 14, "mm_enemy_tower": 14,
                "mm_monster": 8, "mm_buff": 12,
                "mm_ally_minion": 8, "mm_enemy_minion": 8}.get(cls, 20)
        size = int(base * rng.uniform(0.8, 1.25))
        if size < 4 or size >= min(h, w) * 0.5:
            continue
        x = rng.uniform(0, w - size)
        y = rng.uniform(0, h - size)
        paste(cut, img, x, y, size)
        cx = (x + size / 2) / w
        cy = (y + size / 2) / h
        bw = size / w
        bh = size / h
        labels.append((CLS_IDX[cls], cx, cy, bw, bh))
        placed += 1
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-img", type=int, default=8)
    ap.add_argument("--imgs", type=int, default=600)
    ap.add_argument("--out", default="data/mm_dataset")
    ap.add_argument("--bg", default="data/mm_backgrounds_clean",
                    help="背景目录（默认清洗过的：已抹除元素）")
    ap.add_argument("--cutouts", default="data/mm_cutouts_clean",
                    help="抠图目录（默认去背景带 alpha 版）")
    args = ap.parse_args()

    cutout_dir = ROOT / args.cutouts
    bg_dir = ROOT / args.bg
    out_dir = ROOT / args.out
    if not bg_dir.exists():
        print("缺少背景目录 data/mm_backgrounds")
        return

    pool = load_cutouts(cutout_dir)
    if not pool:
        print("缺少抠图目录 data/mm_cutouts/<class>/，请先放入用户提供的抠图")
        return

    bgs = list(bg_dir.glob("*.png"))
    rng = random.Random(42)

    # 清空旧数据
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    n_train = int(args.imgs * 0.8)
    n_val = args.imgs - n_train
    idx = 0
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            bg = cv2.imdecode(np.fromfile(str(rng.choice(bgs)), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
            if bg is None:
                continue
            # 背景本身可能有残留元素（英雄/塔）——合成标签不含它们，训练时当负样本
            img, labels = synthesize_one(bg, pool, rng, args.per_img)
            name = f"{split}_{idx:05d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), img)
            with open(out_dir / "labels" / split / f"{name}.txt", "w") as f:
                for lab in labels:
                    f.write("%d %.4f %.4f %.4f %.4f\n" % lab)
            idx += 1
        print(f"{split}: {n} 张")

    yaml = ("path: %s\ntrain: images/train\nval: images/val\nnames:\n"
            % out_dir.resolve())
    for i, c in enumerate(CLASSES):
        yaml += f"  {i}: {c}\n"
    (out_dir / "data.yaml").write_text(yaml, encoding="utf-8")
    print("完成:", out_dir / "data.yaml")
    print("类别:", CLASSES)


if __name__ == "__main__":
    main()
