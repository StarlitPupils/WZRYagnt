# -*- coding: utf-8 -*-
"""全屏模型特训数据合成器：用户素材（单元素裁剪图）贴到真实对局背景。

用户提供（data/full_extra/<class>/）：
  ally_tower / ally_minion / enemy_minion / enemy_hero / ally_hero

流程：
  1. 背景池：真实对局帧（data/full_bg/ 或 temp/ann/s*.png）+ 用户素材
  2. 每张背景贴 3-8 个元素（随机位置/缩放），生成 YOLO 标注
  3. 与现有 367 张真实标注混合，供 train_11class 重训

用法：
  venv\\Scripts\\python.exe scripts\\train\\build_full_extra.py [--imgs 400]
"""
import argparse
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CLASSES = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
           "enemy_turret", "ally_turret"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
# 类别->configs/classes.json 中的 id
JSON_CLS = {"enemy_hero": 0, "ally_hero": 1, "enemy_minion": 2, "ally_minion": 3,
            "enemy_turret": 4, "ally_turret": 5}


def load_pool(d):
    imgs = []
    if d.exists():
        for f in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            data = np.fromfile(str(f), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs.append(img)
    return imgs


def paste(front, back, x, y, size):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", type=int, default=400)
    ap.add_argument("--bg", default="temp/ann", help="真实对局帧背景目录")
    ap.add_argument("--out", default="data/full_extra_dataset")
    args = ap.parse_args()

    # 目录映射：类别名 -> 用户素材目录名（用户目录叫 ally_tower 而非 ally_turret）
    dir_map = {"ally_turret": "ally_tower"}
    pool = {}
    for c in CLASSES:
        dname = dir_map.get(c, c)
        pool[c] = load_pool(ROOT / "data" / "full_extra" / dname)
    pool = {c: imgs for c, imgs in pool.items() if imgs}
    for c, imgs in pool.items():
        print(f"  {c}: {len(imgs)} 张素材")
    total_pool = sum(len(v) for v in pool.values())
    if total_pool == 0:
        print("无素材")
        return

    bgs = list((ROOT / args.bg).glob("*.png"))
    rng = random.Random(7)
    out_dir = ROOT / args.out
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    n = 0
    n_train = int(args.imgs * 0.8)
    for split in ("train", "val"):
        limit = n_train if split == "train" else args.imgs - n_train
        for _ in range(limit):
            bg = cv2.imdecode(np.fromfile(str(rng.choice(bgs)), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
            if bg is None:
                continue
            img = bg.copy()
            h, w = img.shape[:2]
            labels = []
            n_elem = rng.randint(3, 8)
            for _ in range(n_elem):
                cls = rng.choice(list(pool.keys()))
                src = rng.choice(pool[cls])
                sh, sw = src.shape[:2]
                # 目标尺寸：塔 120-220, 英雄 60-140, 小兵 20-50（相对 720p）
                base = {"enemy_hero": 110, "ally_hero": 110,
                        "enemy_minion": 36, "ally_minion": 36,
                        "enemy_turret": 160, "ally_turret": 160}[cls]
                size = int(base * rng.uniform(0.7, 1.3))
                size = min(size, int(min(h, w) * 0.4))
                if size < 10:
                    continue
                x = rng.uniform(0, w - size)
                y = rng.uniform(0, h - size)
                paste(src, img, x, y, size)
                cx = (x + size / 2) / w
                cy = (y + size / 2) / h
                labels.append(f"{JSON_CLS[cls]} {cx:.4f} {cy:.4f} {size / w:.4f} {size / h:.4f}")
            if not labels:
                continue
            name = f"fx_{n:05d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), img)
            (out_dir / "labels" / split / f"{name}.txt").write_text(
                "\n".join(labels), encoding="utf-8")
            n += 1
        print(f"{split}: {limit} 张")
    print(f"完成 {n} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
