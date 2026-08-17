# -*- coding: utf-8 -*-
import argparse
import json
import shutil
import random
from pathlib import Path


def collect_images(img_dir):
    """收集目录下 (jpg, txt) 对（txt 必须存在）。"""
    pairs = []
    for ext in ['*.jpg', '*.png']:
        for img_path in img_dir.glob(ext):
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                pairs.append((img_path, txt_path))
    return pairs


def prepare_dataset(hero_name, train_ratio=0.8, extra_dirs=None):
    base_dir = Path(f"data/yolo_dataset/{hero_name}")
    img_dir = Path(f"data/screenshots/{hero_name}")

    # 读取类别
    with open("configs/classes.json", "r", encoding="utf-8-sig") as f:
        classes = json.load(f)
    nc = len(classes)

    # 创建目录
    for split in ['train', 'val']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 收集已标注图片（主目录 + 附加目录）
    img_files = collect_images(img_dir)
    for d in (extra_dirs or []):
        img_files += collect_images(Path(d))

    if not img_files:
        print("No annotated images found!")
        return

    random.seed(42)
    random.shuffle(img_files)
    split_idx = int(len(img_files) * train_ratio)
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        for img_path, txt_path in files:
            shutil.copy(img_path, base_dir / 'images' / split / img_path.name)
            shutil.copy(txt_path, base_dir / 'labels' / split / txt_path.name)

    # 生成 data.yaml
    yaml_content = f"""# {hero_name} dataset for YOLOv8
path: {base_dir.absolute().as_posix()}
train: images/train
val: images/val
nc: {nc}
names: {json.dumps(classes, ensure_ascii=False)}
"""
    (base_dir / 'data.yaml').write_text(yaml_content, encoding='utf-8')
    print(f"Dataset prepared: {len(train_files)} train, {len(val_files)} val, {nc} classes.")
    print(f"Config: {base_dir / 'data.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default="zhongkui")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--extra-dir", action="append", default=None,
                    help="附加图片目录（可多次，如 data/screenshots/replay）")
    args = ap.parse_args()
    prepare_dataset(args.hero, args.train_ratio, args.extra_dir)

