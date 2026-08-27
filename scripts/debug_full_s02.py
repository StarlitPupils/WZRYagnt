# -*- coding: utf-8 -*-
"""s02 全屏检测器预测 vs 用户真值对比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector

CLS = {0: "敌方英雄", 1: "我方英雄", 2: "敌方小兵", 3: "我方小兵", 4: "敌方塔",
       5: "我方塔", 6: "敌方水晶", 7: "我方水晶", 8: "野怪", 9: "钩子", 10: "技能"}

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s02.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"), conf=0.25)
print("== 预测 ==")
for d in det.detect(img):
    x1, y1, x2, y2 = (int(v) for v in d.xyxy)
    print(f"  cls={d.cls} ({CLS.get(d.cls,'?')}) conf={d.conf:.2f} box=({x1},{y1})-({x2},{y2})")

print("== GT (s02.txt, 全帧归一化) ==")
for ln in (ROOT / "temp" / "ann" / "s02.txt").read_text(encoding="utf-8").splitlines():
    p = ln.split()
    if len(p) != 5:
        continue
    c = int(p[0])
    if c > 10:
        continue
    x, y, w, h = [float(v) for v in p[1:5]]
    x1, y1 = int((x - w / 2) * 1280), int((y - h / 2) * 720)
    x2, y2 = int((x + w / 2) * 1280), int((y + h / 2) * 720)
    print(f"  cls={c} ({CLS.get(c,'?')}) box=({x1},{y1})-({x2},{y2})")
