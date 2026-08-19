# -*- coding: utf-8 -*-
"""小地图元素 YOLO 强化训练（用户提供抠图合成数据集后运行）。

用法：
  1. 用户抠图放入 data/mm_cutouts/<class>/
  2. python scripts/train/build_mm_dataset.py   # 合成训练集
  3. python scripts/train/train_mm_yolo.py      # 训练
  4. 推理: from wzry.vision.mm_yolo import MMYoloDetector
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mm_dataset/data.yaml")
    ap.add_argument("--name", default="mm_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resume", default=None, help="续训权重路径")
    args = ap.parse_args()

    from ultralytics import YOLO
    data_yaml = ROOT / args.data
    if not data_yaml.exists():
        print("缺少", data_yaml, "，请先运行 build_mm_dataset.py")
        return
    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(str(ROOT / "models" / "yolo" / "yolov8n.pt"))
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=320, batch=16,
                project=str(ROOT / "runs" / "mm_detect"), name=args.name,
                device="0" if __import__("torch").cuda.is_available() else "cpu",
                workers=0, patience=30, exist_ok=True)
    print("训练完成: runs/mm_detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
