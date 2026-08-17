# -*- coding: utf-8 -*-
"""对局状态机：判断当前屏幕处于什么阶段。

v0 启发式（需校准文件提供 minimap_center，归一化坐标）：
  IN_MATCH 判定 = 小地图 ROI 是"暗圆"（平均灰度低） + 存在阵营色圆点（蓝/红）。
  带滞回防抖：连续 N 帧一致才切换状态，避免单帧误判。

局限性（v1 改进方向）：结算画面、选人界面等需单独模板；小地图 ROI 依赖固定分辨率。
"""
import enum
import time

import numpy as np

# 阵营色（BGR），王者荣耀小地图常规配色
_BLUE_LOW = np.array([150, 60, 30])
_BLUE_HIGH = np.array([255, 150, 120])
_RED_LOW = np.array([30, 40, 130])
_RED_HIGH = np.array([110, 120, 255])


class MatchPhase(str, enum.Enum):
    UNKNOWN = "unknown"          # 未识别
    WAITING = "waiting"          # 局外（大厅/房间/加载等），等人工进房
    IN_MATCH = "in_match"        # 对局中
    POST_MATCH = "post_match"    # 结算画面（v1 实现）


class MatchStateMachine:
    def __init__(self, minimap_center_norm, minimap_r_norm=0.082,
                 gray_thr=95.0, min_dots=4, hold_frames=3, post_hold_s=5.0):
        self.minimap_cx, self.minimap_cy = minimap_center_norm
        self.minimap_r_norm = minimap_r_norm
        self.gray_thr = gray_thr
        self.min_dots = min_dots
        self.hold_frames = hold_frames
        self.post_hold_s = post_hold_s   # 结算状态保持秒数，供下游感知"对局结束"

        self.phase = MatchPhase.WAITING
        self._in_match_streak = 0
        self._out_match_streak = 0
        self._post_since = None
        self._mask_cache = None
        self._mask_key = None

    # ---------- 特征 ----------
    def _circle_mask(self, h, w):
        key = (h, w)
        if self._mask_key != key:
            cy = int(self.minimap_cy * h)
            cx = int(self.minimap_cx * w)
            r = max(8, int(self.minimap_r_norm * min(w, h)))
            yy, xx = np.ogrid[:h, :w]
            self._mask_cache = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
            self._mask_key = key
        return self._mask_cache

    def _minimap_features(self, frame):
        h, w = frame.shape[:2]
        mask = self._circle_mask(h, w)
        roi = frame[mask]
        if roi.size == 0:
            return False, 0
        gray = roi.mean()
        blue = np.sum(np.all((roi >= _BLUE_LOW) & (roi <= _BLUE_HIGH), axis=1))
        red = np.sum(np.all((roi >= _RED_LOW) & (roi <= _RED_HIGH), axis=1))
        return gray < self.gray_thr, int(blue) + int(red)

    # ---------- 状态机 ----------
    def update(self, frame):
        """输入一帧 BGR 画面，返回当前 MatchPhase。"""
        if frame is None:
            return self.phase
        dark, dots = self._minimap_features(frame)
        in_match = dark and dots >= self.min_dots

        if in_match:
            self._in_match_streak += 1
            self._out_match_streak = 0
            if self._in_match_streak >= self.hold_frames:
                self.phase = MatchPhase.IN_MATCH
                self._post_since = None
        else:
            self._out_match_streak += 1
            self._in_match_streak = 0
            if self.phase == MatchPhase.IN_MATCH:
                # 对局画面连续消失 → 判定结算，并"粘住" post_hold_s 秒
                if self._out_match_streak >= self.hold_frames:
                    self.phase = MatchPhase.POST_MATCH
                    self._post_since = time.time()
            elif self.phase == MatchPhase.POST_MATCH:
                if self._post_since and time.time() - self._post_since > self.post_hold_s:
                    self.phase = MatchPhase.WAITING
            else:
                self.phase = MatchPhase.WAITING
        return self.phase

    def __repr__(self):
        return f"<MatchStateMachine phase={self.phase.value}>"
