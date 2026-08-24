# -*- coding: utf-8 -*-
"""小地图检测器 v4：HeroCamp CNN 分类英雄圈 + 规则判塔/野怪。

英雄圈 -> 真值训练 3 类 CNN（自己/队友/敌人，val acc 100%）
塔 -> 尺寸+颜色规则（红塔真值 100%命中）
野怪/buff -> 黄色分层
"""
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] if False else "."))
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MM_SIZE = 232
CORNER = 34
BUF = 14
SIZE = 32


def _load_net():
    from scripts.train.train_hero_camp import HeroCampNet
    net = HeroCampNet(3)
    ckpt = ROOT / "runs" / "mm_hero" / "hero_camp.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


class MMDetectorV4:
    def __init__(self):
        self.net = _load_net()

    @torch.no_grad()
    def _camp(self, crop):
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = crop.astype(np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1)[None])
        prob = F.softmax(self.net(x), 1)[0].numpy()
        return int(prob.argmax()), prob[int(prob.argmax())]

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
            if corner(cx, cy):
                continue
            r_ = center_frac(cx, cy, sz, 165, 180) + center_frac(cx, cy, sz, 0, 15)
            y_ = center_frac(cx, cy, sz, 18, 45)
            b_ = center_frac(cx, cy, sz, 70, 135)
            # 黄色优先：野怪点(<=13px)/buff(>=14px)
            if y_ > 0.12:
                if sz >= 14:
                    res["buff"].append((cx, cy))
                else:
                    res["monster"].append((cx, cy))
                continue
            if 8 <= sz <= 22 and r_ > 0.18 and r_ >= b_ * 0.5 and y_ < 0.1:
                res["enemy_tower"].append((cx, cy))
                continue
            if sz < 11:
                if r_ > 0.25:
                    res["enemy_minion"].append((cx, cy))
                else:
                    res["ally_minion"].append((cx, cy))
                continue
            # 英雄圈（sz>=16）: HeroCamp 分类
            if sz >= 16:
                crop = mm[max(0, cy - sz // 2):cy + sz // 2,
                          max(0, cx - sz // 2):cx + sz // 2]
                if crop.size == 0:
                    continue
                camp, conf = self._camp(crop)
                if camp == 0:
                    res["self"].append((cx, cy))
                elif camp == 1:
                    res["ally"].append((cx, cy))
                else:
                    res["enemy"].append((cx, cy))
                continue
            # 11-15px: 塔
            if r_ > 0.2:
                res["enemy_tower"].append((cx, cy))
            else:
                res["ally_tower"].append((cx, cy))

        # 自己唯一化
        if len(res["self"]) > 1:
            scored = []
            for p in res["self"]:
                crop = mm[max(0, p[1] - 12):p[1] + 12, max(0, p[0] - 12):p[0] + 12]
                camp, conf = self._camp(crop) if crop.size else (0, 0)
                scored.append((conf, p))
            res["self"] = [max(scored, key=lambda x: x[0])[1]]

        def norm(p):
            return [round(p[0] / mw, 4), round(p[1] / mh, 4)]

        def rec(pts):
            return [{"n": norm(p), "conf": 0.7, "src": "v4"} for p in pts]

        return {"found": True,
                "center": ((x0 + x1) / 2, (y0 + y1) / 2), "radius": mw / 2,
                "dots": {"self": rec(res["self"]), "ally": rec(res["ally"]),
                         "enemy": rec(res["enemy"]), "monster": rec(res["monster"]),
                         "buff": rec(res["buff"])},
                "towers": {"ally": rec(res["ally_tower"]),
                           "enemy": rec(res["enemy_tower"])},
                "minions": {"ally": rec(res["ally_minion"]),
                            "enemy": rec(res["enemy_minion"])}}
