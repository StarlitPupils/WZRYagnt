# -*- coding: utf-8 -*-
"""全屏 11 类检测全量校验（vs 用户真值 s01-s12，修正前后对比）。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, Det, camp_blue_red_fracs

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"), conf=0.25)
NAME2ID = {n: i for i, n in det.model.names.items()}
NAMES = det.model.names

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

def norm_box(p):
    x, y, w, h = [float(v) for v in p[1:5]]
    return (int((x - w / 2) * 1280), int((y - h / 2) * 720),
            int((x + w / 2) * 1280), int((y + h / 2) * 720))

CLS_CN = ["敌英", "我英", "敌兵", "我兵", "敌塔", "我塔", "敌晶", "我晶", "野怪", "钩子", "技能"]
stats_pre = {i: [0, 0] for i in range(11)}
stats_post = {i: [0, 0] for i in range(11)}
n_gt = n_match = n_skip = 0
fix_log = []
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists() or not gtf.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
    raw = []
    if res.boxes is not None:
        for box, c, cid in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = (float(v) for v in box)
            raw.append((int(cid), [x1, y1, x2, y2]))
    corrected = camp_correct(
        img, [Det(NAMES[c], 0.5, p, [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
              for c, p in raw])
    # 用原始框+修正类
    post_pairs = []
    for (c, p), d in zip(raw, corrected):
        post_pairs.append((NAME2ID[d.cls], p))
    pre_pairs = raw
    gt = []
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c > 10:
            continue
        gt.append((c, norm_box(p)))
    if not gt:
        continue
    for gcls, gbox in gt:
        n_gt += 1
        # 画面中央自己英雄(框中心在 0.3-0.7 且面积>8%) 跳过
        x1, y1, x2, y2 = gbox
        cx, cy = (x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720
        area = (x2 - x1) * (y2 - y1) / (1280 * 720)
        if 0.3 < cx < 0.7 and 0.25 < cy < 0.75 and area > 0.08:
            n_skip += 1
            continue
        best_pre = best_post = None
        for cid_, pbox in pre_pairs:
            v = iou(pbox, gbox)
            if v > 0.3 and (best_pre is None or v > best_pre[0]):
                best_pre = (v, cid_)
        for cid_, pbox in post_pairs:
            v = iou(pbox, gbox)
            if v > 0.3 and (best_post is None or v > best_post[0]):
                best_post = (v, cid_)
        if best_pre is None:
            print(f"s{i:02d} GT={CLS_CN[gcls]} 漏检 box={gbox}")
            continue
        n_match += 1
        pre_c, post_c = best_pre[1], best_post[1]
        stats_pre[gcls][0] += 1
        stats_pre[gcls][1] += int(pre_c == gcls)
        stats_post[gcls][0] += 1
        stats_post[gcls][1] += int(post_c == gcls)
        if post_c != gcls:
            print(f"s{i:02d} GT={CLS_CN[gcls]} 修正后仍错: pred={CLS_CN[post_c]} box={gbox}")
        elif pre_c != gcls:
            fix_log.append(f"s{i:02d} {CLS_CN[gcls]}: {CLS_CN[pre_c]} -> {CLS_CN[post_c]}")

print()
for l in fix_log:
    print("修正生效:", l)
print()
print(f"GT={n_gt} 配准={n_match} 跳过中央自己={n_skip}")
for c in range(11):
    if stats_post[c][0]:
        print(f"  {CLS_CN[c]}: 修正前 {stats_pre[c][1]}/{stats_pre[c][0]}  修正后 {stats_post[c][1]}/{stats_post[c][0]}")
tp = sum(v[1] for v in stats_post.values()); tn = sum(v[0] for v in stats_post.values())
print(f"  合计: 修正前 {sum(v[1] for v in stats_pre.values())}/{tn}  修正后 {tp}/{tn} = {tp/max(1,tn)*100:.0f}%")
