# -*- coding: utf-8 -*-
"""小地图检测器 v5：固定点位学习（用户真值聚类）+ 英雄圈 HeroCamp。

原理（用户标注 12 帧聚类成果）：
  - 蓝塔/红塔/buff/野怪营地 位置固定（每局地图相同）
  - 用户标注出现次数的固定点 = 可靠位置先验
  - 蓝塔与背景同色无法颜色检测 -> 固定点直接判定
  - 其余候选 -> HeroCamp 阵营分类（自己/队友/敌人）

固定点（小地图归一化，来自用户12帧标注聚类）：
  buff: 6 个 | 野怪营地: 11 个 | 蓝塔: 10 个 | 红塔: 10 个
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
MATCH_R = 0.10        # 固定点匹配半径（放宽：候选中心与固定点中心有偏移）

# 固定点（从用户标注聚类，含出现帧数）
BUFF_PTS = [(0.46, 0.21, 8), (0.71, 0.52, 6), (0.53, 0.78, 4),
            (0.28, 0.46, 3), (0.32, 0.29, 2), (0.65, 0.70, 1)]
MONSTER_PTS = [(0.62, 0.20, 10), (0.33, 0.14, 10), (0.18, 0.30, 9),
               (0.52, 0.32, 8), (0.37, 0.79, 8), (0.65, 0.84, 8),
               (0.84, 0.53, 8), (0.78, 0.40, 8), (0.20, 0.58, 7),
               (0.81, 0.68, 6), (0.15, 0.45, 4)]
RED_TOWER_PTS = [(0.52, 0.05, 12), (0.90, 0.09, 12), (0.30, 0.05, 11),
                 (0.74, 0.05, 11), (0.81, 0.18, 11), (0.68, 0.29, 10),
                 (0.95, 0.24, 9), (0.95, 0.43, 9), (0.60, 0.41, 8),
                 (0.95, 0.73, 7)]
BLUE_TOWER_PTS = [(0.03, 0.31, 12), (0.03, 0.56, 12), (0.04, 0.75, 12),
                  (0.10, 0.89, 12), (0.31, 0.70, 12), (0.47, 0.95, 11),
                  (0.24, 0.95, 11), (0.39, 0.58, 11), (0.18, 0.81, 10),
                  (0.73, 0.94, 8)]


def _load_net():
    from scripts.train.train_hero_camp import HeroCampNet
    net = HeroCampNet(3)
    ckpt = ROOT / "runs" / "mm_hero" / "hero_camp.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


def _match_pt(p, pts, thr=MATCH_R):
    """匹配固定点，返回 (类别索引, 置信) 或 None。"""
    best, bd = None, thr
    for x, y, n in pts:
        d = ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5
        if d < bd:
            bd, best = d, (x, y, n)
    return best


class MMDetectorV5:
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
            hh = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            HH, SS = hh[..., 0].astype(int), hh[..., 1].astype(int)
            sat = SS > 60
            return int(((HH >= hlo) & (HH < hhi) & sat).sum()) / max(1, int(sat.sum()))

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
            # 边缘排除（±0.06 内非固定点 = UI/边角元素，HeroCamp 常误判）
            if (n_px[0] < 0.06 or n_px[0] > 0.94 or n_px[1] < 0.06 or n_px[1] > 0.94):
                if not (_match_pt(n_px, RED_TOWER_PTS) or _match_pt(n_px, BLUE_TOWER_PTS)
                        or _match_pt(n_px, MONSTER_PTS) or _match_pt(n_px, BUFF_PTS)):
                    continue
            r_ = center_frac(cx, cy, sz, 165, 180) + center_frac(cx, cy, sz, 0, 15)
            y_ = center_frac(cx, cy, sz, 18, 45)
            b_ = center_frac(cx, cy, sz, 70, 135)
            g_ = center_frac(cx, cy, sz, 35, 90)
            # 自己优先（绿候选，唯一化在最后）
            if sz >= 16 and g_ > 0.15:
                res["self"].append((cx, cy))
                continue
            # 固定点匹配（位置先验）
            if y_ > 0.1:
                hit = _match_pt(n_px, MONSTER_PTS) or _match_pt(n_px, BUFF_PTS)
                if hit:
                    cls = "buff" if hit in BUFF_PTS else "monster"
                    res[cls].append((cx, cy))
                    continue
            if r_ > 0.15 and r_ >= b_ * 0.4 and y_ < 0.1:
                if _match_pt(n_px, RED_TOWER_PTS):
                    continue  # 红塔固定点已直接输出，不重复
            if b_ > 0.5 and sz <= 18:
                if _match_pt(n_px, BLUE_TOWER_PTS):
                    continue  # 蓝塔固定点已直接输出
            # 未匹配固定点：英雄圈（HeroCamp）/小兵
            if sz >= 16:
                if g_ > 0.15:
                    res["self"].append((cx, cy))
                    continue
                crop = mm[max(0, cy - sz // 2):cy + sz // 2,
                          max(0, cx - sz // 2):cx + sz // 2]
                if crop.size == 0:
                    continue
                camp, conf = self._camp(crop)
                if conf >= 0.65:
                    if camp == 0:
                        res["self"].append((cx, cy))
                    elif camp == 1:
                        res["ally"].append((cx, cy))
                    else:
                        res["enemy"].append((cx, cy))
                else:
                    res["ally_minion"].append((cx, cy))
                continue
            if sz < 11:
                if r_ > 0.25:
                    res["enemy_minion"].append((cx, cy))
                else:
                    res["ally_minion"].append((cx, cy))
            else:
                res["ally_minion"].append((cx, cy))

        # 自己唯一化：排除四角泉水区 + 绿占比最高（真值自己绿0.28 vs 队友圈0.08）
        if res["self"]:
            res["self"] = [p for p in res["self"]
                           if not any(abs(p[0] - c[0]) < 26 and abs(p[1] - c[1]) < 26
                                      for c in ((12, 12), (mw - 12, 12),
                                                (12, mh - 12), (mw - 12, mh - 12)))]
        if len(res["self"]) > 1:
            res["self"] = [max(res["self"],
                               key=lambda p: center_frac(p[0], p[1], 20, 35, 90))]

        # 固定点直接输出（蓝塔/红塔/野怪/buff 位置固定，无候选也输出——塔未被摧毁时）
        for x, y, n in BLUE_TOWER_PTS:
            res["ally_tower"].append((int(x * mw), int(y * mh)))
        for x, y, n in RED_TOWER_PTS:
            res["enemy_tower"].append((int(x * mw), int(y * mh)))

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        def rec(pts):
            return [{"n": norm(p), "conf": 0.7, "src": "v5"} for p in pts]

        return {"found": True,
                "center": ((x0 + x1) / 2, (y0 + y1) / 2), "radius": mw / 2,
                "dots": {"self": rec(res["self"]), "ally": rec(res["ally"]),
                         "enemy": rec(res["enemy"]), "monster": rec(res["monster"]),
                         "buff": rec(res["buff"])},
                "towers": {"ally": rec(res["ally_tower"]),
                           "enemy": rec(res["enemy_tower"])},
                "minions": {"ally": rec(res["ally_minion"]),
                            "enemy": rec(res["enemy_minion"])}}
