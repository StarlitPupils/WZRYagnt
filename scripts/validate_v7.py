# -*- coding: utf-8 -*-
"""v7 滑窗环检测 vs 用户真值 (temp/ann/s*.txt 小地图类 11-19)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7

det = MMDetectorV7()

# 真值: 11=自己 12=队友 13=敌人 14=蓝塔 15=红塔 16=野怪 17=buff 18=我兵 19=敌兵
GT = {}
for i in range(1, 13):
    f = ROOT / "temp" / "ann" / f"s{i:02d}.txt"
    if not f.exists():
        continue
    items = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        cls = int(p[0])
        if 11 <= cls <= 19:
            x, y, w, h = [float(v) for v in p[1:5]]
            items.append((cls, (x, y, w, h)))
    GT[i] = items

def norm_minimap(x, y, w, h):
    # 标注为全屏归一化，小地图区域 (0,0)-(232,232)/ (1280,720)
    # 标注工具把小地图点存成全屏归一化 -> 转地图归一化
    px = x * 1280.0
    py = y * 720.0
    return px / 232.0, py / 232.0

def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

total = {"self": [0, 0], "ally": [0, 0], "enemy": [0, 0], "ally_tower": [0, 0], "enemy_tower": [0, 0]}
names = {11: "self", 12: "ally", 13: "enemy", 14: "ally_tower", 15: "enemy_tower"}

out = []
for i in sorted(GT):
    frame_path = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not frame_path.exists():
        continue
    img = cv2.imread(str(frame_path))
    if img is None:
        continue
    # 画布可能是 1280x720 或带标注条
    if img.shape[1] < 1280:
        continue
    canvas = img[:720, :1280]
    r = det.detect(canvas)
    line = f"s{i:02d}: "
    for cls, (x, y, w, h) in sorted(GT[i]):
        key = names.get(cls, None)
        if key is None:
            continue
        gp = norm_minimap(x, y, w, h)
        preds = []
        if key == "self":
            preds = r["dots"]["self"]
        elif key == "ally":
            preds = r["dots"]["ally"]
        elif key == "enemy":
            preds = r["dots"]["enemy"]
        elif key == "ally_tower":
            preds = r["towers"]["ally"]
        elif key == "enemy_tower":
            preds = r["towers"]["enemy"]
        best = min((dist((p["n"][0], p["n"][1]), gp) for p in preds), default=9.0)
        ok = best < 0.12
        total[key][0] += 1
        total[key][1] += int(ok)
        line += f"{key[:4]}@{gp[0]:.2f},{gp[1]:.2f}:{'OK' if ok else f'({best:.2f})'} "
    # 伪检：未标注的检出
    fp_self = 0
    if not any(c == 11 for c, *_ in GT[i]):
        fp_self = len(r["dots"]["self"])
    line += f" FP_self={fp_self}"
    out.append(line)

for l in out:
    print(l)
print()
for k, v in total.items():
    print(f"{k}: {v[1]}/{v[0]} = {v[1]/v[0]*100:.0f}%" if v[0] else f"{k}: no gt")
