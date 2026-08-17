# -*- coding: utf-8 -*-
"""11 类 YOLO 训练脚本（标注到位后一键运行）。

前置：
  1. data/yolo_dataset/zhongkui 下标签已复核/补齐（见 docs/ANNOTATION_GUIDE.md）；
  2. scripts/train/class_stats.py 确认各类框数 > 0（空类会破坏训练）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\train_11class.py [--epochs 120] [--batch 8]
        [--imgsz 640] [--model models/yolo/yolov8n.pt] [--name zhongkui_11cls]
        [--export-trt]     # 训练后导出 TensorRT FP16（需 ultralytics 支持）
        [--resume PATH]    # 从断点恢复
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "data" / "yolo_dataset" / "zhongkui" / "data.yaml"
CLASSES_JSON = ROOT / "configs" / "classes.json"


def check_data(data_dir, allow_empty=False):
    """训练前校验：11 类声明、各类框数、train/val 比例。"""
    with open(CLASSES_JSON, "r", encoding="utf-8-sig") as f:
        classes = json.load(f)
    assert len(classes) == 11, f"classes.json 应为 11 类，实际 {len(classes)}"
    labels = sorted((data_dir / "labels").rglob("*.txt"))
    counts = {}
    for txt in labels:
        for ln in txt.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) == 5:
                cid = int(p[0])
                counts[cid] = counts.get(cid, 0) + 1
    print("=== 11 类框数分布 ===")
    empty = []
    for i, c in enumerate(classes):
        n = counts.get(i, 0)
        print(f"  {i:>2} {c:<16} {n}")
        if n == 0:
            empty.append(c)
    if empty:
        print(f"警告: 以下类别无框（训练会退化）: {empty}")
        if not allow_empty:
            print("建议先按 docs/ANNOTATION_GUIDE.md 补标后再训练；")
            print("实验模式可加 --allow-empty 跳过（空类 mAP 会为 0）。")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo" / "yolov8n.pt"))
    ap.add_argument("--name", default="zhongkui_11cls")
    ap.add_argument("--device", default="0")
    ap.add_argument("--export-trt", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--data", default="zhongkui",
                    help="数据集名（yolo_dataset 下，如 zhongkui / zhongkui_v2）")
    ap.add_argument("--allow-empty", action="store_true",
                    help="实验模式：允许空类训练（空类 mAP 为 0）")
    args = ap.parse_args()

    data_dir = ROOT / "data" / "yolo_dataset" / args.data
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"缺少数据集配置: {data_yaml}")
        return 1
    if not check_data(data_dir, allow_empty=args.allow_empty):
        return 1

    from ultralytics import YOLO

    # 沙盒/受限环境兼容：ultralytics 缓存构建用 multiprocessing.ThreadPool（Windows 命名管道），
    # 在受限环境中被禁止 -> 替换为串行实现；DataLoader 用 workers=0（单进程）。
    try:
        import ultralytics.data.dataset as _ds

        class _SerialPool:
            def __init__(self, *a, **k):
                pass

            def map(self, func, iterable, chunksize=None):
                return [func(x) for x in iterable]

            def imap(self, func, iterable, chunksize=None):
                for x in iterable:
                    yield func(x)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        _ds.ThreadPool = _SerialPool
        print("已启用受限环境兼容模式（串行缓存构建 + workers=0）")
    except Exception as e:
        print(f"兼容模式启用失败（不影响正常环境）: {e}")

    model = YOLO(args.model if args.resume is None else args.resume)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        amp=True,
        workers=0,
    )

    if args.export_trt:
        best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
        print(f"导出 TensorRT FP16: {best}")
        YOLO(str(best)).export(format="engine", half=True, imgsz=args.imgsz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
