# -*- coding: utf-8 -*-
"""全屏英雄框的头顶血条证据: 真(匹配GT) vs FP 的蓝/红条像素数对比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, Det

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls_v3" / "weights" / "best.pt"), conf=0.25)
CLS_CN = ["敌英", "我英"]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

def bar_evidence(img, x1, y1, x2, y2, kind):
    """框上缘 ±25px 内的蓝/红条像素投影: 返回 (条列像素数, 最大宽度)。"""
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(1280, int(x2)), min(720, int(y2))
    y0, y1b = max(0, y1 - 25), min(720, y1 + 25)
    band = img[y0:y1b, x1:x2]
    if band.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    if kind == "ally":
        m = (H >= 90) & (H <= 135) & (S > 55) & (V > 60)
    else:
        m = ((H <= 15) | (H >= 165)) & (S > 60) & (V > 60)
    cols = m.any(axis=0)
    # 最长连续段(允许gap2)
    idx = np.nonzero(cols)[0]
    if len(idx) < 20:
        return 0, 0
    best = cur = 1
    for a, b in zip(idx, idx[1:]):
        cur = cur + 1 if b - a <= 2 else 1
        best = max(best, cur)
    return int(cols.sum()), best

print(f"{'帧':<4}{'类':<4}{'框':<28}{'条列数':>6}{'最长宽度':>7}  匹配GT")
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
    raw = [(int(cid), [float(v) for v in box]) for box, c, cid in zip(
        res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))]
    corr = camp_correct(img, [Det(det.model.names[c], 0.5, p, [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2]) for c, p in raw])
    gt = []
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) > 10:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gt.append((int(p[0]), ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)))
    for (c_, p_), d in zip(raw, corr):
        nid = next((k for k in range(9) if ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion", "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal", "neutral_monster"][k] == d.cls), -1)
        if nid not in (0, 1):
            continue
        matched = any(iou(p_, g) > 0.3 and gc == nid for gc, g in gt)
        kind = "ally" if nid == 1 else "enemy"
        colsum, seg = bar_evidence(img, *p_, kind)
        cx, cy = (p_[0] + p_[2]) / 2 / 1280, (p_[1] + p_[3]) / 2 / 720
        area = (p_[2] - p_[0]) * (p_[3] - p_[1]) / (1280 * 720)
        if nid == 1 and 0.3 < cx < 0.7 and 0.25 < cy < 0.75 and area > 0.05:
            matched = True  # 中央自己
        print(f"s{i:02d} {'我英' if nid == 1 else '敌英':<4}{str([round(v) for v in p_]):<28}{colsum:>6}{seg:>7}  {'是' if matched else '否'}")
