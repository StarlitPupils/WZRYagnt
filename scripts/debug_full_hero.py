# -*- coding: utf-8 -*-
"""每帧原始预测类的分布 + 我方英雄 GT 框内预测内容。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"), conf=0.25)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

from collections import Counter
cnt = Counter()
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
    preds = []
    if res.boxes is not None:
        for box, c, cid in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = (float(v) for v in box)
            preds.append((det.model.names[cid], float(c), [x1, y1, x2, y2]))
            cnt[det.model.names[cid]] += 1
    # GT 我英(1) 框
    for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) != 1:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gbox = ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)
        inside = [(n, c, b) for n, c, b in preds if iou(b, gbox) > 0.1]
        print(f"s{i:02d} GT我英{gbox} 内预测: {[(n, round(c,2)) for n, c, b in inside]}")
print("类别计数:", dict(cnt))
