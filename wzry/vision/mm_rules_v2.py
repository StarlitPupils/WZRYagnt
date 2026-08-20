# -*- coding: utf-8 -*-
"""小地图规则检测器 v2（用户真值标定版）。

依据用户 12 帧标注真值的中心区域颜色/尺寸统计：
  英雄圈(>=19px):  self绿>0.3 | enemy红>0.25且红>=蓝*0.7 | 否则ally
  小兵(<=16px):    red>0.25 -> enemy_minion | 否则 ally_minion
  塔/buff/野怪(11-18px): 黄>0.1 -> buff(>=14)/monster | 红>0.2 -> enemy_tower
                        | 否则 ally_tower
泉水区(四角34px)固定标记排除。
"""
import cv2
import numpy as np

MM_SIZE = 232
CORNER = 34
BUF = 14          # 同类别去重半径


def _comps(mask):
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        out.append((int(cent[i][0]), int(cent[i][1]),
                    int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                    int(st[i, cv2.CC_STAT_AREA])))
    return out


class MMRuleDetectorV2:
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
            return (min(px, mw - px) < CORNER and min(py, mh - py) < CORNER)

        def colors(px, py, sz):
            r = max(2, sz // 2)
            crop = mm[max(0, py - r):py + r, max(0, px - r):px + r]
            if crop.size == 0:
                return 0, 0, 0, 0
            hsv2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            HH, SS, VV = (hsv2[..., 0].astype(int), hsv2[..., 1].astype(int),
                          hsv2[..., 2].astype(int))
            sat = SS > 60
            ns = max(1, int(sat.sum()))
            b = int(((HH >= 90) & (HH <= 135) & sat).sum()) / ns
            r_ = int((((HH <= 15) | (HH >= 165)) & sat).sum()) / ns
            y = int(((HH >= 18) & (HH <= 45) & sat).sum()) / ns
            g = int(((HH >= 35) & (HH <= 90) & sat).sum()) / ns
            return b, r_, y, g

        res = {"self": [], "ally": [], "enemy": [],
               "ally_tower": [], "enemy_tower": [],
               "monster": [], "buff": [],
               "ally_minion": [], "enemy_minion": []}

        # 候选 = 所有高饱和连通域（彩色元素）
        cands = []
        for mask in [(H >= 90) & (H <= 135) & (S > 60) & (V > 60),
                     ((H <= 15) | (H >= 165)) & (S > 60) & (V > 60),
                     (H >= 18) & (H <= 45) & (S > 60) & (V > 60),
                     (H >= 35) & (H <= 90) & (S > 60) & (V > 60)]:
            for cx, cy, ww, hh, area in _comps(mask):
                if area < 15:
                    continue
                cands.append((cx, cy, max(ww, hh)))

        # 去重（同一位置多色连通域合并）
        cands.sort(key=lambda c: -c[2])
        uniq = []
        for c in cands:
            if not any((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 < BUF ** 2 for u in uniq):
                uniq.append(c)

        for cx, cy, sz in uniq:
            if corner(cx, cy):
                continue
            b, r_, y, g = colors(cx, cy, sz)
            if sz >= 19:                      # 英雄圈
                if g > 0.3:
                    res["self"].append((cx, cy))
                elif r_ > 0.25 and r_ >= b * 0.7:
                    res["enemy"].append((cx, cy))
                else:
                    res["ally"].append((cx, cy))
            elif sz <= 16:                    # 小兵
                if r_ > 0.25:
                    res["enemy_minion"].append((cx, cy))
                else:
                    res["ally_minion"].append((cx, cy))
            else:                             # 塔/buff/野怪 (17-18)
                if y > 0.1:
                    if sz >= 14:
                        res["buff"].append((cx, cy))
                    else:
                        res["monster"].append((cx, cy))
                elif r_ > 0.2:
                    res["enemy_tower"].append((cx, cy))
                else:
                    res["ally_tower"].append((cx, cy))

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        def rec(pts):
            return [{"n": norm(p), "conf": 0.7, "src": "rule"} for p in pts]

        return {"found": True,
                "center": ((x0 + x1) / 2, (y0 + y1) / 2), "radius": mw / 2,
                "dots": {"self": rec(res["self"]), "ally": rec(res["ally"]),
                         "enemy": rec(res["enemy"]), "monster": rec(res["monster"]),
                         "buff": rec(res["buff"])},
                "towers": {"ally": rec(res["ally_tower"]),
                           "enemy": rec(res["enemy_tower"])},
                "minions": {"ally": rec(res["ally_minion"]),
                            "enemy": rec(res["enemy_minion"])}}
