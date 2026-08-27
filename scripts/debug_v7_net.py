# -*- coding: utf-8 -*-
"""HeroCampNet 对真/假蓝环样本分类测试。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scripts.train.train_hero_camp import HeroCampNet  # noqa: E402

net = HeroCampNet(3)
net.load_state_dict(torch.load(str(ROOT / "runs" / "mm_hero" / "hero_camp.pt"), map_location="cpu"))
net.eval()

def classify(mm, cx, cy):
    crop = mm[max(0, cy - 16):cy + 16, max(0, cx - 16):cx + 16]
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(net(x), 1)[0].numpy()
    return p

# 真队友 (与GT ally 标注中心): s01 (25,63)(59,92)(102,106)(172,215) s02 (43,25)(81,147)(64,214)
# 假蓝环: s01 (139,92)(135,110) s02 (127,115)
samples = [
    ("true", 1, 25, 63), ("true", 1, 59, 92), ("true", 1, 102, 106), ("true", 1, 172, 215),
    ("true", 2, 43, 25), ("true", 2, 81, 147), ("true", 2, 64, 214),
    ("fake", 1, 139, 92), ("fake", 1, 135, 110), ("fake", 2, 127, 115),
    ("self", 1, 155, 215), ("self", 6, 203, 218),
    ("enemy", 1, 134, 103), ("enemy", 5, 25, 30),
]
NAME = ["自己", "队友", "敌人"]
for tag, i, px, py in samples:
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    p = classify(mm, px, py)
    print(f"{tag:5s} s{i:02d} ({px},{py}): " + " ".join(f"{n}={v:.2f}" for n, v in zip(NAME, p)))
