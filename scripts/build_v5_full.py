# -*- coding: utf-8 -*-
"""v5 数据集扩充: 基础数据集(293) + 用户全屏(54) + 用户小地图(55) -> 16类。

基础数据集的 12 类保持; 用户数据覆盖 16 类(12-15 新小地图类)。
"""
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)
OUT = ROOT / "data" / "yolo_v5_full"


def collect_user():
    full_cls = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                8: 8, 9: 9, 10: 10, 11: 11}
    mm_cls = {12: 12, 13: 13, 14: 14, 15: 15}
    items = []
    for base, cmap, tag in ((ROOT / "data" / "labeling" / "out_mm" / "images", full_cls, "FULL"),
                            (ROOT / "data" / "labeling" / "in_mm", mm_cls, "MM")):
        for imgf in sorted(base.glob("*.png")):
            txt = imgf.with_suffix(".txt")
            if not txt.exists():
                continue
            lines = []
            for ln in txt.read_text(encoding="utf-8").strip().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                t = cmap.get(int(float(p[0])))
                if t is None:
                    continue
                lines.append(f"{t} {' '.join(p[1:])}")
            if lines:
                items.append((imgf, lines, tag))
    return items


def copy_base():
    items = []
    b = ROOT / "data" / "yolo_dataset" / "zhongkui_v2"
    for split in ("train", "val"):
        for imgf in sorted((b / "images" / split).glob("*.*")):
            if imgf.suffix not in (".png", ".jpg", ".jpeg"):
                continue
            lab = b / "labels" / split / imgf.with_suffix(".txt").name
            if lab.exists():
                items.append((imgf, lab.read_text(encoding="utf-8").strip().splitlines(), "BASE"))
    return items


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    items = copy_base() + collect_user()
    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.15))
    val_items = items[:n_val]
    train_items = items[n_val:]
    n = m = 0
    for imgf, lines, tag in train_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        name = f"{tag}_{imgf.stem}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{name}.png"), img)
        (OUT / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    for imgf, lines, tag in val_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        name = f"{tag}_{imgf.stem}"
        cv2.imwrite(str(OUT / "images" / "val" / f"{name}.png"), img)
        (OUT / "labels" / "val" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        m += 1
    print(f"v5_full: train {n} / val {m} (共{len(items)})")
    names = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
             "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
             "neutral_monster", "hook_aim", "skill_effect", "self",
             "mm_red", "mm_blue", "mm_green", "mm_yellow"]
    yaml_p = OUT / "data.yaml"
    yaml_p.write_text(
        "path: " + str(OUT).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")
    print("data.yaml 就绪")


if __name__ == "__main__":
    main()
