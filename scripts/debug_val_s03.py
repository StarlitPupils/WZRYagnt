# -*- coding: utf-8 -*-
"""复现 validate_full2 对 s03 的匹配逻辑，找出漏匹配原因。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, hero_dedup_ui, Det

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls_v4" / "weights" / "best.pt"), conf=0.25)
names = det.model.names
CLS_ID = {0: "enemy_hero", 1: "ally_hero", 2: "enemy_minion", 3: "ally_minion",
          4: "enemy_turret", 5: "ally_turret", 6: "enemy_crystal", 7: "ally_crystal",
          8: "neutral_monster", 9: "hook_aim", 10: "skill_effect"}

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s03.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
raw = [(int(cid), float(c), [float(v) for v in box]) for box, c, cid in zip(
    res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))]
corrected = hero_dedup_ui(camp_correct(
    img, [Det(names[c], cf, p, [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2]) for c, cf, p in raw]))
post_pairs = []
for (c_, cf_, p_), d in zip(raw, corrected):
    nid = next(k for k, v in CLS_ID.items() if v == d.cls)
    post_pairs.append((nid, p_))
print("管线后预测:")
for nid, p_ in post_pairs:
    print("  ", nid, [round(v) for v in p_])
gt = []
for ln in (ROOT / "temp" / "ann" / "s03.txt").read_text(encoding="utf-8").splitlines():
    p = ln.split()
    if len(p) != 5 or int(p[0]) > 10:
        continue
    x, y, w, h = [float(v) for v in p[1:5]]
    gt.append((int(p[0]), ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)))
print("GT:")
for gid, g in gt:
    b = None
    for nid, p_ in post_pairs:
        v = iou(p_, g)
        if v > 0.3 and (b is None or v > b[0]):
            b = (v, nid)
    print("  ", gid, [round(v) for v in g], "->", b)
