# -*- coding: utf-8 -*-
"""小地图混合检测器 v7：YOLO(塔/self/buff/monster) + 颜色圈(英雄) 交叉验证。

原理（用户语义）：
  - 小地图塔 = 固定位置方块（YOLO mm_ally_tower/mm_enemy_tower 可靠）
  - 英雄标记 = 圆圈+箭头+头像（颜色圈：绿=自己/蓝=队友/红=敌人）
  - 固定位置的红/蓝小方块是塔，不是英雄 —— 用 YOLO 塔位置扣除颜色圈

流程：
  1. YOLO v6 检测（self/towers/monster/buff/minions）
  2. 颜色检测圈（蓝/红，排除四角泉水区）
  3. 颜色圈 与 YOLO 塔位置去重 -> 英雄候选（蓝=队友 红=敌人）
"""
import cv2
import numpy as np

from wzry.vision.mm_yolo import MMYoloDetector

MM_SIZE = 232
CORNER_MARGIN = 30      # 四角泉水固定标记排除半径
TOWER_RADIUS = 14       # 与 YOLO 塔框去重距离


class MMHybridDetector:
    """小地图混合检测：输出 self/ally/enemy/towers/monster/buff/minions。"""

    def __init__(self, yolo_weights=None, yolo_conf=0.3):
        self.yolo = MMYoloDetector(yolo_weights, conf=yolo_conf)

    # ---------- 颜色圈 ----------
    def _color_circles(self, mm):
        """英雄圈检测（尺寸区分塔/野怪/buff）：
        英雄圈 22-32px(area 250-700)，塔方块 10-16px(area 60-180)，
        野怪点 6-10px，buff 图标 12-18px。
        """
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        H, S, V = (hsv[..., 0].astype(int), hsv[..., 1].astype(int),
                   hsv[..., 2].astype(int))
        out = {"blue": [], "red": []}
        for name, m in [("blue", (H >= 90) & (H <= 135) & (S > 80) & (V > 80)),
                        ("red", ((H <= 12) | (H >= 168)) & (S > 80) & (V > 80))]:
            mask = m.astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                area = int(st[i, cv2.CC_STAT_AREA])
                w_ = int(st[i, cv2.CC_STAT_WIDTH])
                h_ = int(st[i, cv2.CC_STAT_HEIGHT])
                comp = area / max(1.0, w_ * h_)
                # 英雄圈：大尺寸(>=20px 宽) 且近圆
                if not (w_ >= 20 and h_ >= 20 and area >= 250
                        and comp >= 0.55):
                    continue
                cx, cy = int(cent[i][0]), int(cent[i][1])
                # 排除四角泉水固定标记
                if min(cx, MM_SIZE - cx) < CORNER_MARGIN and \
                        min(cy, MM_SIZE - cy) < CORNER_MARGIN:
                    continue
                out[name].append((cx, cy))
        return out

    @staticmethod
    def _near(pt, pts, thr):
        for q in pts:
            if (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 < thr * thr:
                return True
        return False

    def detect(self, frame, mm_box=None):
        """返回与 mm_yolo.detect 同构的 dict，ally/enemy 用颜色圈+塔去重。"""
        r = self.yolo.detect(frame, mm_box)
        if not r.get("found"):
            return r
        h, w = frame.shape[:2]
        if mm_box is None:
            mm_box = (0, 0, min(MM_SIZE, w), min(MM_SIZE, h))
        x0, y0, x1, y1 = mm_box
        mm = frame[y0:y1, x0:x1]

        # YOLO 塔位置（小地图像素）
        ally_tw = [(t["n"][0] * (x1 - x0), t["n"][1] * (y1 - y0))
                   for t in r["towers"]["ally"]]
        enemy_tw = [(t["n"][0] * (x1 - x0), t["n"][1] * (y1 - y0))
                    for t in r["towers"]["enemy"]]
        all_tw = ally_tw + enemy_tw

        # 颜色圈 -> 英雄（与塔去重）
        circles = self._color_circles(mm)
        r["dots"]["ally"] = [
            {"n": [round(c[0] / (x1 - x0), 4), round(c[1] / (y1 - y0), 4)],
             "conf": 0.6, "src": "color"}
            for c in circles["blue"] if not self._near(c, all_tw, TOWER_RADIUS)]
        r["dots"]["enemy"] = [
            {"n": [round(c[0] / (x1 - x0), 4), round(c[1] / (y1 - y0), 4)],
             "conf": 0.6, "src": "color"}
            for c in circles["red"] if not self._near(c, all_tw, TOWER_RADIUS)]
        # self：绿圈（大尺寸）颜色验证优先，YOLO self 兜底
        green = self._color_green(mm)
        if green:
            r["dots"]["self"] = [
                {"n": [round(c[0] / (x1 - x0), 4), round(c[1] / (y1 - y0), 4)],
                 "conf": 0.7, "src": "green"} for c in green]
        return r

    def _color_green(self, mm):
        """自己绿圈（大尺寸近圆）检测。"""
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        H, S, V = (hsv[..., 0].astype(int), hsv[..., 1].astype(int),
                   hsv[..., 2].astype(int))
        m = ((H >= 35) & (H <= 90) & (S > 80) & (V > 80)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
        out = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            w_ = int(st[i, cv2.CC_STAT_WIDTH])
            h_ = int(st[i, cv2.CC_STAT_HEIGHT])
            comp = area / max(1.0, w_ * h_)
            if w_ >= 20 and h_ >= 20 and area >= 250 and comp >= 0.55:
                out.append((int(cent[i][0]), int(cent[i][1])))
        return out
