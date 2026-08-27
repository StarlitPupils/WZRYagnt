# -*- coding: utf-8 -*-
"""1) 全屏各阵营预测的 FP 率(vs 真值 IoU 匹配不上的预测框)
   2) 小地图 ally/enemy 检出的位置偏差分布。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, Det
from wzry.vision.mm_rules_v7 import MMDetectorV7

det = YoloDetector(str(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "detect" / "zhongkui_11cls_v3" / "weights" / "best.pt"), conf=0.25)
mmdet = MMDetectorV7()
CLS_CN = ["敌英", "我英", "敌兵", "我兵", "敌塔", "我塔", "敌晶", "我晶", "野怪", "钩子", "技能"]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

print("=== 1) 全屏 FP 分析 (预测框无 GT 匹配) ===")
fp_stat = {}
gt_boxes = {}
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    # 用完整管线: det.detect() 已含 camp_correct + hero_dedup_ui
    dets = det.detect(img)
    post = []
    for d in dets:
        nid = next((k for k in range(9) if ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion", "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal", "neutral_monster"][k] == d.cls), -1)
        post.append((nid, d.xyxy, d.conf))
    gt = []
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) > 10:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gt.append((int(p[0]), ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)))
    gt_boxes[i] = gt
    for cid_, pbox, conf in post:
        # 中央自己特例
        cx, cy = (pbox[0] + pbox[2]) / 2 / 1280, (pbox[1] + pbox[3]) / 2 / 720
        area = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) / (1280 * 720)
        matched = any(iou(pbox, g) > 0.3 and gc == cid_ for gc, g in gt)
        if cid_ == 1 and 0.3 < cx < 0.7 and 0.25 < cy < 0.75 and area > 0.05:
            continue  # 中央自己
        fp_stat.setdefault(cid_, 0)
        fp_stat[cid_] += 0 if matched else 1
        if not matched and cid_ in (0, 1):
            print(f"s{i:02d} 未匹配 {CLS_CN[cid_]} box={[round(v) for v in pbox]}")
print("各阵营 FP 计数:", {CLS_CN[k]: v for k, v in sorted(fp_stat.items())})

print()
print("=== 2) 小地图位置偏差 (检出 vs 真值, 匹配半径内) ===")
import collections
errs = collections.defaultdict(list)
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    r = mmdet.detect(img[:720, :1280])
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in (11, 12, 13):
            continue
        x, y = float(p[1]), float(p[2])
        px, py = x * 1280 / 232, y * 720 / 232
        key = {11: "self", 12: "ally", 13: "enemy"}[c]
        cands = r["dots"][key]
        if not cands:
            continue
        best = min(cands, key=lambda d: ((d["n"][0] - px) ** 2 + (d["n"][1] - py) ** 2) ** 0.5)
        d = ((best["n"][0] * 232 - px * 232) ** 2 + (best["n"][1] * 232 - py * 232) ** 2) ** 0.5
        if d < 30:
            errs[key].append(d)
for k, v in errs.items():
    v = np.array(v)
    print(f"{k}: n={len(v)} 平均误差={v.mean():.1f}px 中位={np.median(v):.1f}px p90={np.percentile(v, 90):.1f}px 最大={v.max():.1f}px")
