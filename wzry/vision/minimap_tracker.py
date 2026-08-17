# -*- coding: utf-8 -*-
"""小地图跟踪器：利用"小地图在 UI 上位置固定"的特性做快速持续定位。

策略：
  - 首帧（或丢失后）：以校准先验圆心为种子做全扫描（find_minimap 完整模式）；
  - 后续帧：固定圆心 + 半径扫描（fast 模式），单帧 30-100ms；
  - 圆点结构门控：英雄点蓝/红各 ≤5 且至少 1 个，否则判失联（防 UI 圆圈假阳性）；
  - 连续丢失 N 帧：重新全扫描（防 UI 变化/分辨率切换）。
"""
import time

import cv2
import numpy as np

from wzry.vision import minimap


class MinimapTracker:
    def __init__(self, prior_center=None, found_thr=None, relost_after=15,
                 min_radius_frac=0.09, max_radius_frac=0.22):
        self.prior_center = list(prior_center) if prior_center else None
        self.found_thr = found_thr or minimap.FOUND_SCORE_MIN
        self.relost_after = relost_after
        self.min_radius_frac = min_radius_frac
        self.max_radius_frac = max_radius_frac

        self.center = None
        self.radius = None
        self.lost_streak = 0
        self.last_result = None
        self.last_ms = 0.0

    # ---------- 内部工具 ----------
    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        masks_f = {k: v.astype(np.float32) for k, v in minimap._color_masks(frame).items()}
        return gray_f, masks_f

    def _scan_around(self, gray_f, masks_f, cx, cy, r_center, r_lo, r_hi):
        """固定圆心扫描半径 + 圆心微调，返回 (score, cx, cy, r) 或 None。"""
        best = None
        for r in range(max(r_lo, r_center - 10), min(r_hi, r_center + 11), 2):
            for ddy in (-4, 0, 4):
                for ddx in (-4, 0, 4):
                    m = minimap._disk_metrics(gray_f, masks_f, cx + ddx, cy + ddy, r)
                    if m is None:
                        continue
                    s = minimap._score_disk(m)
                    if best is None or s > best[0]:
                        best = (s, cx + ddx, cy + ddy, r)
        return best

    def _plausible(self, frame, center, radius):
        """圆点结构门控：英雄点蓝/红各 1-5 个才可信（排除 UI 圆圈/场景误检）。"""
        fake = {"found": True, "center": center, "radius": radius}
        det = minimap.detect_dots(frame, fake)
        n_blue = len(det["dots"]["blue"])
        n_red = len(det["dots"]["red"])
        return (1 <= n_blue <= 5 and 1 <= n_red <= 5), det

    # ---------- 主入口 ----------
    def update(self, frame):
        """输入 BGR 帧，返回 analyze 风格结果；内部维护中心与半径。"""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        short = min(h, w)
        r_lo = max(16, int(short * self.min_radius_frac))
        r_hi = max(40, int(short * self.max_radius_frac))
        gray_f, masks_f = self._preprocess(frame)

        det = None
        if self.center is not None and self.lost_streak < self.relost_after:
            # 跟踪模式
            best = self._scan_around(gray_f, masks_f,
                                     self.center[0], self.center[1],
                                     self.radius, r_lo, r_hi)
            if best and best[0] >= self.found_thr:
                ok, det = self._plausible(frame, [best[1], best[2]], best[3])
                if ok:
                    self.center, self.radius = [best[1], best[2]], best[3]
                    self.lost_streak = 0
                else:
                    self.lost_streak += 1
            else:
                self.lost_streak += 1
        else:
            # 全扫描模式（种子 = 先验圆心）
            res = minimap.find_minimap(frame, prior=self.prior_center)
            if res["found"]:
                ok, det = self._plausible(frame, res["center"], res["radius"])
                if ok:
                    self.center, self.radius = res["center"], res["radius"]
                    self.lost_streak = 0
                else:
                    self.lost_streak += 1
            else:
                self.lost_streak += 1

        self.last_ms = (time.perf_counter() - t0) * 1000.0
        if self.center is None:
            self.last_result = {"found": False, "center": None, "radius": None,
                                "method": None,
                                "dots": {"blue": [], "red": [], "yellow": []},
                                "towers": [], "detail": {}}
            return self.last_result

        if det is None:
            fake = {"found": True, "center": self.center, "radius": self.radius}
            det = minimap.detect_dots(frame, fake)
        self.last_result = {"found": True, "center": self.center, "radius": self.radius,
                            "method": "track",
                            "dots": det["dots"], "towers": det["towers"],
                            "detail": det["detail"]}
        return self.last_result
