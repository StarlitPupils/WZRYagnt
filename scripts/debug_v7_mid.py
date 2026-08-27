# -*- coding: utf-8 -*-
"""s01 中部区域放大 + GT 标记。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
# 真值(文件) 全帧归一化 -> 地图px: px=x*1280, 是地图px (小地图区 0-232)
# 敌人(0.58,0.48)->(135,111); 队友(0.44,0.47)->(102,109); 队友(0.26,0.40)->(60,93)
crop = mm[70:130, 100:180]
big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
# 标记敌人 GT (135,111) -> (135-100,111-70)*8=(280,328)
cv2.circle(big, (280, 328), 14, (0, 255, 255), 3)   # 黄=GT敌
# 队友GT (102,109) -> (16,312)
cv2.circle(big, (16, 312), 14, (255, 0, 255), 3)    # 洋红=GT队友
# 检出敌 (134,103) -> (272,264)
cv2.circle(big, (272, 264), 14, (0, 0, 0), 3)       # 黑=检出敌
# 检出蓝 (139,92) -> (312,176)
cv2.circle(big, (312, 176), 14, (0, 255, 0), 3)     # 绿=检出蓝
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s01_mid.png"), big)
print("saved")
