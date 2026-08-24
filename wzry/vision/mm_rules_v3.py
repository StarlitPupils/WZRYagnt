# -*- coding: utf-8 -*-
"""小地图组合检测器 v3（真值标定最优组合）。

策略（依据用户真值 12 帧评估）：
  红塔:     颜色规则（红方块，100/100 命中）
  自己/敌人: 真值模板最近邻（self 100% / enemy 79%）
  蓝系:     尺寸分层（ally 圈>=19 / 蓝塔 11-18 / 我兵<=10）
  野怪/buff: 黄色占比分层
"""
import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MM_SIZE = 232
CORNER = 34
BUF = 14

TMPL_DIR = ROOT / "data" / "mm_templates_dir"
HIST_BINS = (24, 16)
HIST_SIZE = 32


def _hist(img):
    img = cv2.resize(img, (HIST_SIZE, HIST_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


class MMDetectorV3:
    def __init__(self):
        self.tmpl = {}
        for cls in ("mm_self", "mm_ally", "mm_enemy", "mm_ally_tower",
                    "mm_enemy_tower", "mm_monster", "mm_buff",
                    "mm_ally_minion", "mm_enemy_minion"):
            fs = sorted(glob.glob(str(TMPL_DIR / cls / "*.png")))
            feats = [_hist(cv2.imread(f)) for f in fs[:20] if cv2.imread(f) is not None]
            if feats:
                self.tmpl[cls] = feats
        self._hist_cache = {}

    def _classify(self, crop):
        f = _hist(crop)
        best, bs = None, 0.0
        for cls, feats in self.tmpl.items():
            s = max(cv2.compareHist(f, t, cv2.HISTCMP_CORREL) for t in feats[:8])
            if s > bs:
                bs, best = s, cls
        return best, bs

    def detect(self, frame, mm_box=None):
        h, w = frame.shape[:2]
        if mm_box is None:
            mm_box = (0, 0, min(MM_SIZE, w), min(MM_SIZE, h))
        x0, y0, x1, y1 = mm_box
        mm = frame[y0:y1, x0:x1]
        mw, mh = x1 - x0, y1 - y0
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        H, S, V = (hsv[..., 0].astype(int), hsv[..., 1].astype(int),
                   hsv[..., 2].astype(int))

        def corner(px, py):
            return min(px, mw - px) < CORNER and min(py, mh - py) < CORNER

        def comps(mask):
            mask = mask.astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
            return [(int(cent[i][0]), int(cent[i][1]),
                     int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                     int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n)]

        def center_colors(px, py, sz):
            r = max(2, sz // 2)
            crop = mm[max(0, py - r):py + r, max(0, px - r):px + r]
            if crop.size == 0:
                return 0, 0, 0, 0
            hh = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            HH, SS, VV = (hh[..., 0].astype(int), hh[..., 1].astype(int),
                          hh[..., 2].astype(int))
            sat = SS > 60
            ns = max(1, int(sat.sum()))
            return (int(((HH >= 70) & (HH <= 135) & sat).sum()) / ns,
                    int((((HH <= 15) | (HH >= 165)) & sat).sum()) / ns,
                    int(((HH >= 18) & (HH <= 45) & sat).sum()) / ns,
                    int(((HH >= 35) & (HH <= 90) & sat).sum()) / ns)

        res = {k: [] for k in ("self", "ally", "enemy", "ally_tower",
                               "enemy_tower", "monster", "buff",
                               "ally_minion", "enemy_minion")}

        cands = []
        for mask in [(H >= 70) & (H <= 135) & (S > 50) & (V > 40),
                     ((H <= 15) | (H >= 165)) & (S > 60) & (V > 60),
                     (H >= 18) & (H <= 45) & (S > 60) & (V > 60),
                     (H >= 35) & (H <= 90) & (S > 60) & (V > 60)]:
            for cx, cy, ww, hh, area in comps(mask):
                if area >= 15:
                    cands.append((cx, cy, max(ww, hh)))
        cands.sort(key=lambda c: -c[2])
        uniq = []
        for c in cands:
            if not any((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 < BUF ** 2 for u in uniq):
                uniq.append(c)

        for cx, cy, sz in uniq:
            if corner(cx, cy):
                continue
            b, r_, y, g = center_colors(cx, cy, sz)
            r_hi = r_ > 0.18 and r_ >= b * 0.5
            # 红塔规则（优先，可靠）
            if 8 <= sz <= 22 and r_hi and y < 0.1:
                res["enemy_tower"].append((cx, cy))
                continue
            # 自己（绿）
            if g > 0.3 and sz >= 19:
                res["self"].append((cx, cy))
                continue
            # 敌人（红主导）
            if r_hi and sz >= 19:
                res["enemy"].append((cx, cy))
                continue
            # 野怪/buff（黄）
            if y > 0.1:
                if sz >= 14:
                    res["buff"].append((cx, cy))
                else:
                    res["monster"].append((cx, cy))
                continue
            # 蓝系：尺寸分层（队友圈>=19 / 蓝塔 11-18 / 我兵<=10）
            if sz >= 19:
                res["ally"].append((cx, cy))
            elif sz >= 11:
                res["ally_tower"].append((cx, cy))
            else:
                res["ally_minion"].append((cx, cy))

        # 自己唯一化：只保留绿占比最高的候选
        if len(res["self"]) > 1:
            scored = []
            for p in res["self"]:
                b2, r2, y2, g2 = center_colors(p[0], p[1], 20)
                scored.append((g2, p))
            res["self"] = [max(scored, key=lambda x: x[0])[1]]

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        def rec(pts):
            return [{"n": norm(p), "conf": 0.7, "src": "v3"} for p in pts]

        return {"found": True,
                "center": ((x0 + x1) / 2, (y0 + y1) / 2), "radius": mw / 2,
                "dots": {"self": rec(res["self"]), "ally": rec(res["ally"]),
                         "enemy": rec(res["enemy"]), "monster": rec(res["monster"]),
                         "buff": rec(res["buff"])},
                "towers": {"ally": rec(res["ally_tower"]),
                           "enemy": rec(res["enemy_tower"])},
                "minions": {"ally": rec(res["ally_minion"]),
                            "enemy": rec(res["enemy_minion"])}}
