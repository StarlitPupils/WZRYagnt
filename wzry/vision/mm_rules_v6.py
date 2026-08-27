# -*- coding: utf-8 -*-
"""小地图检测器 v6：环状结构（英雄圈）判定。

决定性特征（真值验证）：
  英雄圈 = 空心环（外圈色+中心头像非色）：中心30%区域目标色占比 ~0.00
  地图绿块/泉水 = 实心：中心占比 0.23-0.56
  塔/野怪/buff = 实心小点（固定点位）

流程：
  1. 颜色候选 -> 中心占比 < 0.10 = 环状 -> HeroCamp 分类(自己/队友/敌人)
  2. 实心候选 -> 匹配固定点(塔/野怪/buff) -> 输出；不匹配忽略
  3. 塔按固定点直接输出（背景同色元素用位置先验）
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MM_SIZE = 232
SIZE = 32
BUF = 14
MATCH_R = 0.10
RING_FRAC = 0.10       # 中心占比 < 0.10 = 环状（英雄圈）

BUFF_PTS = [(0.46, 0.21), (0.71, 0.52), (0.53, 0.78),
            (0.28, 0.46), (0.32, 0.29), (0.65, 0.70)]
MONSTER_PTS = [(0.62, 0.20), (0.33, 0.14), (0.18, 0.30), (0.52, 0.32),
               (0.37, 0.79), (0.65, 0.84), (0.84, 0.53), (0.78, 0.40),
               (0.20, 0.58), (0.81, 0.68), (0.15, 0.45)]
RED_TOWER_PTS = [(0.52, 0.05), (0.90, 0.09), (0.30, 0.05), (0.74, 0.05),
                 (0.81, 0.18), (0.68, 0.29), (0.95, 0.24), (0.95, 0.43),
                 (0.60, 0.41), (0.95, 0.73)]
BLUE_TOWER_PTS = [(0.03, 0.31), (0.03, 0.56), (0.04, 0.75), (0.10, 0.89),
                  (0.31, 0.70), (0.47, 0.95), (0.24, 0.95), (0.39, 0.58),
                  (0.18, 0.81), (0.73, 0.94)]


def _load_net():
    from scripts.train.train_hero_camp import HeroCampNet
    net = HeroCampNet(3)
    ckpt = ROOT / "runs" / "mm_hero" / "hero_camp.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


def _near(p, pts, thr=MATCH_R):
    for x, y in pts:
        if ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5 < thr:
            return True
    return False


class MMDetectorV6:
    def __init__(self):
        self.net = _load_net()

    @torch.no_grad()
    def _camp(self, crop):
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = crop.astype(np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1)[None])
        prob = F.softmax(self.net(x), 1)[0].numpy()
        i = int(prob.argmax())
        return i, float(prob[i])

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
            return [(int(cent[i][0]), int(cent[i][1]),
                     int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                     int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n)]

        def center_frac(px, py, sz, hlo, hhi):
            r = max(2, sz // 2)
            crop = mm[max(0, py - r):py + r, max(0, px - r):px + r]
            if crop.size == 0:
                return 0
            hh, ww = crop.shape[:2]
            hsv2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            HH, SS = hsv2[..., 0].astype(int), hsv2[..., 1].astype(int)
            sat = SS > 60
            m = ((HH >= hlo) & (HH < hhi) & sat)
            cy0, cy1 = int(hh * 0.35), int(hh * 0.65)
            cx0, cx1 = int(ww * 0.35), int(ww * 0.65)
            c_frac = float(m[cy0:cy1, cx0:cx1].sum()) / max(1.0, (cy1 - cy0) * (cx1 - cx0))
            return c_frac

        res = {k: [] for k in ("self", "ally", "enemy", "ally_tower",
                               "enemy_tower", "monster", "buff",
                               "ally_minion", "enemy_minion")}

        cands = []
        for mask in [(H >= 70) & (H <= 135) & (S > 60) & (V > 60),
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
            n_px = (cx / mw, cy / mh)
            # 边缘排除（非固定点）
            if (n_px[0] < 0.06 or n_px[0] > 0.94 or n_px[1] < 0.06 or n_px[1] > 0.94):
                if not (_near(n_px, RED_TOWER_PTS) or _near(n_px, BLUE_TOWER_PTS)
                        or _near(n_px, MONSTER_PTS) or _near(n_px, BUFF_PTS)):
                    continue
            # 主色相
            r_ = center_frac(cx, cy, sz, 165, 180) + center_frac(cx, cy, sz, 0, 15)
            b_ = center_frac(cx, cy, sz, 70, 135)
            g_ = center_frac(cx, cy, sz, 35, 90)
            y_ = center_frac(cx, cy, sz, 18, 45)
            main = max([('r', r_), ('b', b_), ('g', g_), ('y', y_)], key=lambda x: x[1])
            # 环状检查（主色相）：中心<0.10 且外圈>0.15
            if sz >= 16:
                hlo, hhi = {'r': (165, 180), 'b': (70, 135), 'g': (35, 90),
                            'y': (18, 45)}[main[0]]
                cfrac = center_frac(cx, cy, sz, hlo, hhi)
                if cfrac < RING_FRAC:
                    # 外圈存在（环外目标色）
                    r = max(2, sz // 2)
                    crop = mm[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                    if crop.size:
                        hsv2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        HH, SS = hsv2[..., 0].astype(int), hsv2[..., 1].astype(int)
                        tot = float(((HH >= hlo) & (HH < hhi) & (SS > 60)).sum())
                        if tot > 80:
                            # 英雄圈：直接按主色相判阵营（绿=自己/蓝=队友/红=敌人）
                            if main[0] == 'g':
                                res["self"].append((cx, cy))
                            elif main[0] == 'r' and r_ > b_ * 1.2:
                                res["enemy"].append((cx, cy))
                            elif main[0] == 'b':
                                res["ally"].append((cx, cy))
                            elif main[0] == 'r':
                                res["enemy"].append((cx, cy))
                            continue
            # 实心候选：匹配固定点（塔/野怪/buff）
            if _near(n_px, MONSTER_PTS):
                res["monster"].append((cx, cy))
            elif _near(n_px, BUFF_PTS):
                res["buff"].append((cx, cy))
            elif 8 <= sz <= 22 and _near(n_px, RED_TOWER_PTS):
                res["enemy_tower"].append((cx, cy))
            elif 8 <= sz <= 22 and _near(n_px, BLUE_TOWER_PTS):
                res["ally_tower"].append((cx, cy))

        # 塔固定点直接输出（位置先验）
        for x, y in BLUE_TOWER_PTS:
            res["ally_tower"].append((int(x * mw), int(y * mh)))
        for x, y in RED_TOWER_PTS:
            res["enemy_tower"].append((int(x * mw), int(y * mh)))

        # 自己唯一化：排除四角泉水 + 绿占比最高
        if res["self"]:
            res["self"] = [p for p in res["self"]
                           if not any(abs(p[0] - c[0]) < 26 and abs(p[1] - c[1]) < 26
                                      for c in ((12, 12), (mw - 12, 12),
                                                (12, mh - 12), (mw - 12, mh - 12)))]
        if len(res["self"]) > 1:
            res["self"] = [max(res["self"],
                               key=lambda p: center_frac(p[0], p[1], 20, 35, 90))]

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        def rec(pts):
            return [{"n": norm(p), "conf": 0.7, "src": "v6"} for p in pts]

        return {"found": True,
                "center": ((x0 + x1) / 2, (y0 + y1) / 2), "radius": mw / 2,
                "dots": {"self": rec(res["self"]), "ally": rec(res["ally"]),
                         "enemy": rec(res["enemy"]), "monster": rec(res["monster"]),
                         "buff": rec(res["buff"])},
                "towers": {"ally": rec(res["ally_tower"]),
                           "enemy": rec(res["enemy_tower"])},
                "minions": {"ally": rec(res["ally_minion"]),
                            "enemy": rec(res["enemy_minion"])}}
