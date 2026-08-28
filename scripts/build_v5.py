# -*- coding: utf-8 -*-
"""v5 检测数据集: 全屏(12类) + 小地图(4类 mm_red/blue/green/yellow) 合并 -> 16类。

类别映射 (YOLO 检测):
  0 enemy_hero 1 ally_hero 2 enemy_minion 3 ally_minion
  4 enemy_turret 5 ally_turret 6 enemy_crystal 7 ally_crystal
  8 neutral_monster 9 hook_aim 10 skill_effect 11 self
  12 mm_red      13 mm_blue     14 mm_green     15 mm_yellow
"""
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)
OUT = ROOT / "data" / "yolo_v5"


def collect(base_dir, cls_map, tag):
    """从标注目录收集 (img, lines)。 cls_map: 原始类->目标类(None=跳过)。"""
    items = []
    for imgf in sorted(base_dir.glob("*.png")):
        txt = imgf.with_suffix(".txt")
        if not txt.exists():
            continue
        lines = []
        for ln in txt.read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            t = cls_map.get(c)
            if t is None:
                continue
            lines.append(f"{t} {' '.join(p[1:])}")
        if lines:
            items.append((imgf, lines, tag))
    return items


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    full_cls = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                8: 8, 9: 9, 10: 10, 11: 11}
    mm_cls = {12: 12, 13: 13, 14: 14, 15: 15}

    items = []
    items += collect(ROOT / "data" / "labeling" / "out_mm" / "images", full_cls, "FULL")
    items += collect(ROOT / "data" / "labeling" / "in_mm", mm_cls, "MM")
    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.2))
    val_items = items[:n_val]
    train_items = items[n_val:]
    n = 0
    for imgf, lines, tag in train_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        name = f"{tag}_{imgf.stem}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{name}.png"), img)
        (OUT / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    m = 0
    for imgf, lines, tag in val_items:
        img = cv2.imread(str(imgf))
        if img is None:
            continue
        name = f"{tag}_{imgf.stem}"
        cv2.imwrite(str(OUT / "images" / "val" / f"{name}.png"), img)
        (OUT / "labels" / "val" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        m += 1
    print(f"v5 数据集: train {n} / val {m} (共{len(items)})")
    # data.yaml
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
