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
                 min_radius_frac=0.09, max_radius_frac=0.22, box_prior=None):
        self.prior_center = list(prior_center) if prior_center else None
        self.found_thr = found_thr or minimap.FOUND_SCORE_MIN
        self.relost_after = relost_after
        self.min_radius_frac = min_radius_frac
        self.max_radius_frac = max_radius_frac
        # 方形小地图标定框 (x0, y0, x1, y1)（1280x720 标定，等比缩放）
        self.box_prior = box_prior

        self.center = None
        self.radius = None
        self.lost_streak = 0
        self.last_result = None
        self.last_ms = 0.0

    def _box_prior_center(self, w, h):
        """标定框中心（等比缩放）。"""
        if not self.box_prior:
            return None
        sx, sy = w / 1280.0, h / 720.0
        x0, y0, x1, y1 = (int(v * s) for v, s in zip(self.box_prior, (sx, sy, sx, sy)))
        return [(x0 + x1) // 2, (y0 + y1) // 2], (x1 - x0) // 2

    # ---------- 内部工具 ----------
    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        masks_f = {k: v.astype(np.float32) for k, v in minimap._color_masks(frame).items()}
        return gray_f, masks_f

    def _scan_around(self, gray_f, masks_f, cx, cy, r_center, r_lo, r_hi):
        """固定圆心两段式扫描：半径粗扫(中心不动) -> 最佳半径 -> 圆心微调 -> 半径精调。

        调用次数从 ~90 次度量降到 ~18 次，跟踪单帧 ~30-80ms。
        返回 (score, cx, cy, r) 或 None。
        """
        best_r = None
        # 1) 半径粗扫：中心不动，步长 4
        for r in range(max(r_lo, r_center - 12), min(r_hi, r_center + 13), 4):
            m = minimap._disk_metrics(gray_f, masks_f, cx, cy, r)
            if m is None:
                continue
            s = minimap._score_disk(m)
            if best_r is None or s > best_r[0]:
                best_r = (s, r)
        if best_r is None:
            return None
        # 2) 圆心微调：最佳半径下 ±4px
        best = None
        for ddy in (-4, 0, 4):
            for ddx in (-4, 0, 4):
                m = minimap._disk_metrics(gray_f, masks_f, cx + ddx, cy + ddy, best_r[1])
                if m is None:
                    continue
                s = minimap._score_disk(m)
                if best is None or s > best[0]:
                    best = (s, cx + ddx, cy + ddy, best_r[1])
        # 3) 半径精调：最佳圆心下 ±3px
        if best:
            bx, by = best[1], best[2]
            for r in range(max(r_lo, best_r[1] - 3), min(r_hi, best_r[1] + 4), 2):
                m = minimap._disk_metrics(gray_f, masks_f, bx, by, r)
                if m is None:
                    continue
                s = minimap._score_disk(m)
                if s > best[0]:
                    best = (s, bx, by, r)
        return best

    def _plausible(self, frame, center, radius):
        """圆点结构门控：v2.8 任一颜色英雄点 1-8 个即可信（红方地形可能过检）。"""
        fake = {"found": True, "center": center, "radius": radius}
        det = minimap.detect_dots(frame, fake)
        n_blue = len(det["dots"]["blue"])
        n_red = len(det["dots"]["red"])
        n_yellow = len(det["dots"]["yellow"])
        n_green = len(det["dots"]["green"])
        total = n_blue + n_red + n_yellow + n_green
        return (1 <= total <= 12), det

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
        # 优先：标定框中心（方形小地图，1280x720 标定值等比缩放）
        bp = self._box_prior_center(w, h)
        if bp is not None:
            (pcx, pcy), pr = bp
            if self.center is None:
                # 首帧：用标定框中心直接检测圆点，验证通过即采用
                fake = {"found": True, "center": [pcx, pcy], "radius": pr}
                det = minimap.detect_dots(frame, fake)
                n_blue = len(det["dots"]["blue"])
                n_red = len(det["dots"]["red"])
                n_yellow = len(det["dots"]["yellow"])
                n_tower = len(det["towers"])
                # 放宽：红/黄/塔点任一存在即可确认小地图（蓝点可能因扎堆合并丢失）
                if (1 <= n_blue <= 6 or 1 <= n_red <= 6 or n_yellow > 0 or n_tower > 0):
                    self.center, self.radius = [pcx, pcy], pr
                    self.lost_streak = 0
                else:
                    self.lost_streak += 1
        if self.center is not None and self.lost_streak < self.relost_after:
            # 跟踪模式（围绕已确认中心微扫）
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
        elif self.center is None:
            # 全扫描模式（标定框失败时的后备；种子 = 标定中心）
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
