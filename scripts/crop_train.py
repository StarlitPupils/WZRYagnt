# -*- coding: utf-8 -*-
"""抠图训练 (用户明确: 训练的是"抠出来的图", 不是坐标回归 !)

流程:
  ① 读用户标注 (YOLO txt: cls cx cy w h)
  ② 按框抠出 ROI 小图 (保留原图比例, 四周扩边 10%)
  ③ 组织为 ImageFolder 数据集: data/retrain_crops/{cls_name}/*.png
  ④ 训练小分类器 (迁移 yolo8n 骨干 or 轻量 CNN) -> models/roi_cls.pt
  ⑤ 推理: roi_cls.predict(crop) -> 类别(id) ; 供 yolo 候选消歧/野怪拒绝

用法:
  python scripts/crop_train.py --crop       # 从标注抠图 -> datasets
  python scripts/crop_train.py --train      # 训练分类器
  python scripts/crop_train.py --predict <img>  # 单图测
"""
import argparse
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIRS = ["data/labeling/in_mm", "data/labeling/out_mm/images"]
CROP_ROOT = ROOT / "data" / "retrain_crops"
CLASSES = [
    "enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
    "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
    "neutral_monster", "hook_aim", "skill_effect", "self",
    "mm_red", "mm_blue", "mm_green", "mm_yellow", "mm_monster", "mm_tower",
]


def crop_all(min_side=16, max_side=512):
    """读所有标注, 抠 ROI -> data/retrain_crops/{cls}/*.png。"""
    stats = {}
    total = 0
    for ld in LABEL_DIRS:
        d = ROOT / ld
        for img_p in sorted(d.glob("*.png")):
            txt = img_p.with_suffix(".txt")
            if not txt.exists():
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
            for i, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                # 扩边 10%
                px, py = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                x1, y1 = max(0, x1 - px), max(0, y1 - py)
                x2, y2 = min(w, x2 + px), min(h, y2 + py)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0 or min(roi.shape[:2]) < min_side:
                    continue
                # 等比例缩放到 max_side
                rh, rw = roi.shape[:2]
                scale = min(1.0, max_side / max(rh, rw))
                if scale < 1.0:
                    roi = cv2.resize(roi, (int(rw * scale), int(rh * scale)),
                                     interpolation=cv2.INTER_AREA)
                cls_name = CLASSES[cls] if cls < len(CLASSES) else f"cls{cls}"
                out_d = CROP_ROOT / cls_name
                out_d.mkdir(parents=True, exist_ok=True)
                out_p = out_d / f"{img_p.stem}_{i}.png"
                cv2.imwrite(str(out_p), roi)
                stats[cls_name] = stats.get(cls_name, 0) + 1
                total += 1
    print(f"抠图完成: {total} 张 -> data/retrain_crops/")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")


def train(epochs=25, imgsz=128):
    """轻量 CNN 分类器 (无 torch 依赖版: 朴素 CNN via numpy 太慢;
    优先 ultralytics 分类模型 yolo8n-cls; 无则提示安装)。"""
    import os
    try:
        from ultralytics import YOLO
    except ImportError:
        print("需要 ultralytics: pip install ultralytics")
        return
    n = sum(1 for _ in CROP_ROOT.glob("*/**/*.png"))
    print(f"数据集 {n} 张, 训练 yolo8n-cls (epochs={epochs})")
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(CROP_ROOT), epochs=epochs, imgsz=imgsz,
                project=str(ROOT / "runs" / "cls"), name="roi_cls", exist_ok=True)
    print("训练完成: runs/cls/roi_cls/weights/best.pt")


def predict(img_path, model_path="runs/cls/roi_cls/weights/best.pt", topk=3):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("需要 ultralytics")
        return
    import torch
    model = YOLO(str(ROOT / model_path))
    r = model.predict(str(img_path), verbose=False)[0]
    probs = r.probs
    top = probs.top5 if hasattr(probs, "top5") else None
    names = model.names
    print("预测结果:")
    if top is not None:
        for i, c in enumerate(top):
            print(f"  {names[c]}: {probs.top5conf[i]:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", metavar="IMG")
    args = ap.parse_args()
    if args.crop:
        crop_all()
    elif args.train:
        train()
    elif args.predict:
        predict(args.predict)
    else:
        ap.print_help()
        print("默认执行 --crop (抠图)")
        crop_all()
