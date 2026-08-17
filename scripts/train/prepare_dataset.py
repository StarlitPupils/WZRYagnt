# -*- coding: utf-8 -*-
import json
import shutil
import random
from pathlib import Path

def prepare_dataset(hero_name, train_ratio=0.8):
    base_dir = Path(f"data/yolo_dataset/{hero_name}")
    img_dir = Path(f"data/screenshots/{hero_name}")
    
    # 读取类别
    with open("configs/classes.json", "r") as f:
        classes = json.load(f)
    nc = len(classes)
    
    # 创建目录
    for split in ['train', 'val']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 收集已标注图片
    img_files = []
    for ext in ['*.jpg', '*.png']:
        for img_path in img_dir.glob(ext):
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                img_files.append((img_path, txt_path))
    
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
names: {json.dumps(classes)}
"""
    (base_dir / 'data.yaml').write_text(yaml_content, encoding='utf-8')
    print(f"Dataset prepared: {len(train_files)} train, {len(val_files)} val, {nc} classes.")
    print(f"Config: {base_dir / 'data.yaml'}")

if __name__ == "__main__":
    prepare_dataset("zhongkui")
