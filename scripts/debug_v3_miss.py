# -*- coding: utf-8 -*-
"""新模型: 我英/我兵/我塔 漏检清单。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, Det

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls_v3" / "weights" / "best.pt"), conf=0.25)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists() or not gtf.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
    raw = [(int(cid), [float(v) for v in box]) for box, c, cid in zip(
        res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))]
    corr = camp_correct(img, [Det(det.model.names[c], 0.5, p, [(p[0]+p[2])/2, (p[1]+p[3])/2]) for c, p in raw])
    post = []
    for (c_, p_), d in zip(raw, corr):
        nid = [k for k, v in {0:"enemy_hero",1:"ally_hero",2:"enemy_minion",3:"ally_minion",4:"enemy_turret",5:"ally_turret"}.items() if v == d.cls]
        post.append((nid[0] if nid else -1, p_))
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) not in (1, 3, 5):
            continue
        c = int(p[0])
        x, y, w, h = [float(v) for v in p[1:5]]
        gbox = ((x-w/2)*1280, (y-h/2)*720, (x+w/2)*1280, (y+h/2)*720)
        cx, cy = (gbox[0]+gbox[2])/2/1280, (gbox[1]+gbox[3])/2/720
        area = (gbox[2]-gbox[0])*(gbox[3]-gbox[1])/(1280*720)
        if c == 1 and 0.3 < cx < 0.7 and 0.25 < cy < 0.75 and area > 0.08:
            continue
        best = None
        for cid_, pbox in post:
            v = iou(pbox, gbox)
            if v > 0.3 and (best is None or v > best[0]):
                best = (v, cid_)
        if best is None or best[1] != c:
            print(f"s{i:02d} GT cls={c} box={gbox} 漏检; 附近预测={[(cid_, [round(v) for v in pb]) for cid_, pb in post if iou(pb, gbox) > 0.05][:4]}")
