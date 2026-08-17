# -*- coding: utf-8 -*-
"""Pre-label unlabeled zhongkui screenshots with the finetuned 4-class YOLOv8
detector, writing YOLO txt files in the 11-class target label space.

Skips images that already have a same-name .txt next to the .jpg.

Class mapping (model 4 classes -> 11-class index):
    enemy_hero -> 0
    hook_aim   -> 9
    minion     -> 2 (enemy_minion; may actually be an ally minion, needs human review)
    turret     -> 4 (enemy_turret; may actually be an ally turret, needs human review)

Usage:
    python prelabel.py [--images data/screenshots/zhongkui] [--conf 0.35] [--dry-run]
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

CLASS_MAP = {0: 0, 1: 9, 2: 2, 3: 4}  # model class id -> 11-class id


def parse_args():
    p = argparse.ArgumentParser(description="Pre-label unlabeled zhongkui screenshots.")
    p.add_argument("--images", type=Path,
                   default=Path("data/screenshots/zhongkui"),
                   help="Directory containing screenshots (*.jpg).")
    p.add_argument("--weights", type=Path,
                   default=Path("runs/detect/zhongkui_detector_finetune/weights/best.pt"),
                   help="Path to YOLOv8 weights.")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold.")
    p.add_argument("--device", default="0",
                   help="Inference device (0, cpu, ...).")
    p.add_argument("--batch", type=int, default=16,
                   help="Inference batch size (keep modest; 8GB VRAM).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only count what would be written; do not write files.")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = args.images
    jpgs = sorted(img_dir.glob("*.jpg"))
    todo = [p for p in jpgs if not p.with_suffix(".txt").exists()]
    skipped = len(jpgs) - len(todo)
    print(f"Total jpg: {len(jpgs)} | already labeled (skipped): {skipped} | to prelabel: {len(todo)}")

    if not todo:
        print("Nothing to prelabel.")
        return

    model = YOLO(str(args.weights))
    per_class = Counter()
    total_boxes = 0
    labeled_imgs = 0
    written = 0

    def run_inference(device):
        return model.predict(
            source=[str(p) for p in todo],
            conf=args.conf, iou=args.iou, device=device,
            batch=args.batch, imgsz=args.imgsz, verbose=False,
        )

    try:
        results = run_inference(args.device)
    except Exception as exc:  # e.g. CUDA unavailable
        if args.device != "cpu":
            print(f"[warn] device '{args.device}' failed ({exc}); falling back to cpu")
            results = run_inference("cpu")
        else:
            raise

    for img_path, res in zip(todo, results):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        labeled_imgs += 1
        lines = []
        for cls_id, (cx, cy, w, h) in zip(boxes.cls.tolist(), boxes.xywhn.tolist()):
            new_id = CLASS_MAP.get(int(round(cls_id)))
            if new_id is None:
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            per_class[new_id] += 1
            total_boxes += 1
            lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if lines and not args.dry_run:
            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text("".join(lines), encoding="utf-8")
            written += 1

    names = {0: "enemy_hero", 2: "enemy_minion", 4: "enemy_turret", 9: "hook_aim"}
    print(f"Prelabeled images (>=1 box): {labeled_imgs}/{len(todo)}")
    if not args.dry_run:
        print(f"Txt files written: {written}")
    print(f"Total boxes: {total_boxes}")
    for cid in sorted(per_class):
        print(f"  class {cid} ({names.get(cid, '?')}): {per_class[cid]}")
    if args.dry_run:
        print("[dry-run] no files written")


if __name__ == "__main__":
    main()
