# -*- coding: utf-8 -*-
"""血条左侧等级徽章特征测量: 真血条 vs 伤害数字/红圈弧。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def badge_feat(img, x0, y, tag):
    """徽章区 (x0-36..x0-6, y-8..y+18) 暗像素/白像素占比。"""
    xa, xb = max(0, x0 - 36), x0 - 6
    ya, yb = max(0, y - 8), y + 18
    patch = img[ya:yb, xa:xb]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2].astype(int)
    S = hsv[..., 1].astype(int)
    dark = float((V < 70).mean())
    white = float(((V > 170) & (S < 80)).mean())
    print(f"{tag}: dark={dark:.2f} white={white:.2f}")
    return dark, white

# 真敌英血条: s03 红条 (1174,242,w=110) -> x0=1119
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s03.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
badge_feat(img, 1119, 242, "s03 敌英血条(真)")
# 真我英血条: s01 (804,257,w~92) -> x0=758
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
badge_feat(img, 758, 250, "s01 我英血条(真)")
# 伪: v4_f05 伤害数字 868 (bar y≈340, x0≈617)
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "pseudo1" / "images" / "v4_f05.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
badge_feat(img, 617, 336, "v4f05 伤害数字868(伪)")
# 伪: v4_f05 红圈弧 (E box 下沿) bar y≈325? 用 box x0=617 大致位置
