# -*- coding: utf-8 -*-
"""真实标注数据增强：12 张用户标注 -> 增强变体（翻转/亮度/缩放/平移）。

输出 data/mm_dataset_v8/（纯真实 + 增强，无合成数据）
"""
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MM_W, MM_H = 232, 232


def aug(img, labels, rng):
    """返回增强后的 (img, labels)。labels: [(c, cx, cy, bw, bh)] 归一化。"""
    h, w = img.shape[:2]
    # 水平翻转
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        labels = [(c, 1 - cx, cy, bw, bh) for c, cx, cy, bw, bh in labels]
    # 亮度/对比度
    a = rng.uniform(0.7, 1.3)
    b = rng.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)
    # 小平移（±10px）
    dx = rng.randint(-10, 10)
    dy = rng.randint(-10, 10)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    labels = [(c, (cx * w + dx) / w, (cy * h + dy) / h, bw, bh)
              for c, cx, cy, bw, bh in labels]
    # 缩放抖动（0.9-1.1）
    s = rng.uniform(0.9, 1.1)
    nw, nh = int(w * s), int(h * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, max(0, (h - nh) // 2), max(0, (h - nh + 1) // 2),
                             max(0, (w - nw) // 2), max(0, (w - nw + 1) // 2),
                             cv2.BORDER_REFLECT)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img, labels


def main():
    src = ROOT / "data" / "mm_user_v7"
    out = ROOT / "data" / "mm_dataset_v8"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    n = 0
    for f in sorted((src / "images" / "train").glob("*.png")):
        lbl = src / "labels" / "train" / (f.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        labels = []
        for ln in lbl.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) == 5:
                labels.append((int(p[0]), float(p[1]), float(p[2]),
                               float(p[3]), float(p[4])))
        # 原图 -> val
        cv2.imwrite(str(out / "images" / "val" / f.name), img)
        (out / "labels" / "val" / (f.stem + ".txt")).write_text(
            "\n".join(f"{c} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}"
                      for c, cx, cy, bw, bh in labels), encoding="utf-8")
        # 20 个增强变体 -> train
        for k in range(20):
            ai, al = aug(img.copy(), labels.copy(), rng)
            name = f"{f.stem}_a{k}.png"
            cv2.imwrite(str(out / "images" / "train" / name), ai)
            (out / "labels" / "train" / (Path(name).stem + ".txt")).write_text(
                "\n".join(f"{c} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}"
                          for c, cx, cy, bw, bh in al), encoding="utf-8")
            n += 1
    nt = len(list((out / "images" / "train").glob("*")))
    nv = len(list((out / "images" / "val").glob("*")))
    print(f"v8: train={nt} val={nv}")
    yaml = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
    for i, c in enumerate(["mm_self", "mm_ally", "mm_enemy", "mm_ally_tower",
                           "mm_enemy_tower", "mm_monster", "mm_buff",
                           "mm_ally_minion", "mm_enemy_minion"]):
        yaml += f"  {i}: {c}\n"
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print("data.yaml 就绪")


if __name__ == "__main__":
    main()
