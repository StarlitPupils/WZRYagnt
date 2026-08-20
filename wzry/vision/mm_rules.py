# -*- coding: utf-8 -*-
"""小地图规则检测器 v9：颜色+尺寸+位置规则（用户真值标定）。

依据用户标注真值（12 帧）：
  英雄圈: 19-38px（均值28），蓝=队友/红=敌人/绿=自己
  塔: 10-16px 方块（蓝=我方/红=敌方）
  野怪点: 6-10px 黄点；buff: 12-18px 黄块
  四角泉水区固定标记排除
"""
import cv2
import numpy as np

MM_SIZE = 232
CORNER = 34          # 泉水区排除半径（含基地标记）
BUF_RADIUS = 18      # 塔-圈去重半径


class MMRuleDetector:
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

        def comps(mask):
            mask = mask.astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
            out = []
            for i in range(1, n):
                area = int(st[i, cv2.CC_STAT_AREA])
                ww = int(st[i, cv2.CC_STAT_WIDTH])
                hh = int(st[i, cv2.CC_STAT_HEIGHT])
                comp = area / max(1.0, ww * hh)
                out.append((int(cent[i][0]), int(cent[i][1]), ww, hh, area, comp))
            return out

        def corner(p):
            cx, cy = p
            return (min(cx, mw - cx) < CORNER and min(cy, mh - cy) < CORNER)

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        # ---- 塔（10-16px 方块，低紧凑度或小尺寸）----
        towers_ally = []
        towers_enemy = []
        for cx, cy, ww, hh, area, comp in comps((H >= 90) & (H <= 135) & (S > 70) & (V > 70)):
            if 8 <= ww <= 20 and 8 <= hh <= 20 and 40 <= area <= 260:
                if not corner((cx, cy)):
                    towers_ally.append((cx, cy))
        for cx, cy, ww, hh, area, comp in comps(((H <= 12) | (H >= 168)) & (S > 70) & (V > 70)):
            if 8 <= ww <= 20 and 8 <= hh <= 20 and 40 <= area <= 260:
                if not corner((cx, cy)):
                    towers_enemy.append((cx, cy))

        def near(p, pts):
            return any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < BUF_RADIUS ** 2
                       for q in pts)

        # ---- 英雄圈（>=18px 大圈）----
        self_pts, ally_pts, enemy_pts = [], [], []
        for cx, cy, ww, hh, area, comp in comps((H >= 35) & (H <= 90) & (S > 75) & (V > 75)):
            if ww >= 18 and hh >= 18 and area >= 200 and comp >= 0.5 and not corner((cx, cy)):
                self_pts.append((cx, cy))
        for cx, cy, ww, hh, area, comp in comps((H >= 90) & (H <= 135) & (S > 75) & (V > 75)):
            if ww >= 18 and hh >= 18 and area >= 200 and comp >= 0.5 and not corner((cx, cy)):
                if not near((cx, cy), towers_ally):
                    ally_pts.append((cx, cy))
        for cx, cy, ww, hh, area, comp in comps(((H <= 12) | (H >= 168)) & (S > 75) & (V > 75)):
            if ww >= 18 and hh >= 18 and area >= 200 and comp >= 0.5 and not corner((cx, cy)):
                if not near((cx, cy), towers_enemy):
                    enemy_pts.append((cx, cy))

        # ---- 野怪点(6-10px) / buff(12-18px) 黄色 ----
        monsters, buffs = [], []
        for cx, cy, ww, hh, area, comp in comps((H >= 18) & (H <= 45) & (S > 75) & (V > 75)):
            if 5 <= ww <= 11 and 5 <= hh <= 11:
                monsters.append((cx, cy))
            elif 12 <= ww <= 20 and 12 <= hh <= 20:
                buffs.append((cx, cy))

        def to_rec(pts):
            return [{"n": norm(p), "conf": 0.6, "src": "rule"} for p in pts]

        return {
            "found": True,
            "center": ((x0 + x1) / 2, (y0 + y1) / 2),
            "radius": mw / 2,
            "dots": {"self": to_rec(self_pts), "ally": to_rec(ally_pts),
                     "enemy": to_rec(enemy_pts), "monster": to_rec(monsters),
                     "buff": to_rec(buffs)},
            "towers": {"ally": to_rec(towers_ally), "enemy": to_rec(towers_enemy)},
            "minions": {"ally": [], "enemy": []},
        }
