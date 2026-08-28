# -*- coding: utf-8 -*-
"""YOLO 检测模型强化训练 (v4.3): 用用户标注素材微调 zhongkui_11cls.

组合策略:
  base = 原 yolo 检测模型 (runs/detect/zhongkui_11cls/weights/best.pt)
  新训练集 = 原大数据集 data/yolo_dataset/zhongkui_v2 + 用户新标注 data/yolo_v4
  类别对齐 (共用原 12 类): enemy_hero0 ally_hero1 enemy_minion2 ally_minion3
   enemy_turret4 ally_turret5 enemy_crystal6 ally_crystal7 neutral_monster8
   hook_aim9 skill_effect10 self11

用法:
  python scripts/train_detector_v4.py --epochs 40 --imgsz 640
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def merge_datasets():
    """合并原数据集(zhongkui_v2) + 新标注(yolo_v4) -> data/yolo_v4_merged。"""
    import random
    random.seed(42)
    out = ROOT / "data" / "yolo_v4_merged"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # 清空旧
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        for f in (out / sub).glob("*"):
            f.unlink()

    def copy_set(src_root, tag, split):
        src_img = src_root / "images" / split
        src_lab = src_root / "labels" / split
        if not src_img.exists():
            return 0
        n = 0
        exts = ("*.png", "*.jpg", "*.jpeg")
        fs = []
        for e in exts:
            fs += list(src_img.glob(e))
        for f in fs:
            lab = src_lab / f.with_suffix(".txt").name
            if lab.exists():
                dst_img = out / "images" / split / f"{tag}_{f.name}"
                dst_lab = out / "labels" / split / f"{tag}_{lab.name}"
                shutil.copy2(f, dst_img)
                shutil.copy2(lab, dst_lab)
                n += 1
        return n

    n1 = copy_set(ROOT / "data" / "yolo_dataset" / "zhongkui_v2", "b", "train")
    n2 = copy_set(ROOT / "data" / "yolo_v4", "u", "train")
    n3 = copy_set(ROOT / "data" / "yolo_dataset" / "zhongkui_v2", "b", "val")
    n4 = copy_set(ROOT / "data" / "yolo_v4", "u", "val")
    print(f"合并: train base{n1}+user{n2}, val base{n3}+user{n4}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    data_dir = ROOT / "data" / "yolo_v4_merged"
    if not args.skip_merge:
        merge_datasets()

    # data.yaml
    yaml_p = data_dir / "data.yaml"
    yaml_p.write_text(
        f'path: {str(data_dir).replace(chr(92), "/")}\n'
        'train: images/train\nval: images/val\nnc: 12\n'
        'names: ["enemy_hero","ally_hero","enemy_minion","ally_minion",'
        '"enemy_turret","ally_turret","enemy_crystal","ally_crystal",'
        '"neutral_monster","hook_aim","skill_effect","self"]\n',
        encoding="utf-8")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("需要 ultralytics")
        return
    base = str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt")
    model = YOLO(base)
    print(f"从 {base} 微调, epochs={args.epochs} imgsz={args.imgsz}")
    model.train(data=str(yaml_p), epochs=args.epochs, imgsz=args.imgsz,
                project=str(ROOT / "runs" / "detect"), name="zhongkui_v4",
                exist_ok=True, batch=8, patience=30)
    print("完成: runs/detect/zhongkui_v4/weights/best.pt")


if __name__ == "__main__":
    main()
