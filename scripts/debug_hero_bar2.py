# -*- coding: utf-8 -*-
"""框顶 45px 内 红/蓝血条段 vs 英雄真值阵营。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls_v3" / "weights" / "best.pt"), conf=0.25)

def bar_segs(img, x1, y1, x2, y2, kind):
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(1280, int(x2)), min(720, int(y2))
    band = img[max(0, y1 - 20):y1 + 45, x1:x2]
    if band.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    if kind == "blue":
        m = (H >= 90) & (H <= 135) & (S > 55) & (V > 60)
    else:
        m = ((H <= 15) | (H >= 165)) & (S > 60) & (V > 60)
    cols = m.any(axis=0)
    idx = np.nonzero(cols)[0]
    if len(idx) < 15:
        return 0, 0
    best = cur = 1
    for a, b in zip(idx, idx[1:]):
        cur = cur + 1 if b - a <= 2 else 1
        best = max(best, cur)
    return int(cols.sum()), best

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

for i in (1, 3, 5, 7, 8, 9, 10, 11):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
    gt = []
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) > 10:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gt.append((int(p[0]), ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)))
    for box, c, cid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
        if cid not in (0, 1):
            continue
        p = [float(v) for v in box]
        bs, bs_seg = bar_segs(img, *p, "blue")
        rs, rs_seg = bar_segs(img, *p, "red")
        matched = [gc for gc, g in gt if iou(p, g) > 0.3]
        tag = f"GT={matched}" if matched else "无GT"
        print(f"s{i:02d} {det.model.names[cid]:<11} conf={float(c):.2f} 蓝条={bs_seg:>3} 红条={rs_seg:>3}  {tag}")
