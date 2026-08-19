# -*- coding: utf-8 -*-
"""对局状态机：判断当前屏幕处于什么阶段。

v2.13（真机实测修复）判定逻辑：
  IN_MATCH = 小地图方形 ROI（左上角 0..0.181W x 0..0.322H，1280x720 标定）
     平均灰度低（暗） + 绿色像素多（小地图绿色地图底）+ 蓝色像素不过多。

背景：
  - v0 用"圆形 ROI 暗 + 蓝/红点计数"，个人主页导航栏（深色+大面积蓝色）误判为对局，
    agent 在主页上继续执行触摸（危险）。圆形 ROI 半径仅 59px 也漏检边缘绿点。
  - 修复：方形 ROI 覆盖整个小地图；对局必有绿色地图底（green>50），
    主页/选人界面无绿色小地图（green≈0）；主页/选人蓝色导航栏 blue>1900 超上限排除。

带滞回防抖：连续 N 帧一致才切换状态，避免单帧误判。
"""
import enum
import time

import cv2
import numpy as np

# 阵营色（BGR），王者荣耀小地图常规配色
_BLUE_LOW = np.array([150, 60, 30])
_BLUE_HIGH = np.array([255, 150, 120])
_RED_LOW = np.array([30, 40, 130])
_RED_HIGH = np.array([110, 120, 255])

# 小地图方形区域（归一化，1280x720 标定：0..232px）
_MM_X_NORM = 0.181
_MM_Y_NORM = 0.322

# v2.13 判定阈值
_GRAY_THR = 110.0     # ROI 平均灰度低于此 = 暗
_GREEN_THR = 50       # ROI 内绿色像素（HSV H35-90,S>80,V>90）> 此 = 有绿色地图底
_BLUE_MAX = 1000      # ROI 内蓝色像素 < 此（主页/选人蓝色导航栏 >1900）


class MatchPhase(str, enum.Enum):
    UNKNOWN = "unknown"          # 未识别
    WAITING = "waiting"          # 局外（大厅/房间/加载等），等人工进房
    IN_MATCH = "in_match"        # 对局中
    POST_MATCH = "post_match"    # 结算画面


class MatchStateMachine:
    def __init__(self, minimap_center_norm=None, minimap_r_norm=0.082,
                 gray_thr=_GRAY_THR, min_dots=3, hold_frames=5, post_hold_s=5.0,
                 green_thr=_GREEN_THR, blue_max=_BLUE_MAX):
        # minimap_center_norm 保留兼容旧调用（不再用于 ROI 定位，ROI 固定方形）
        self.gray_thr = gray_thr
        self.green_thr = green_thr
        self.blue_max = blue_max
        self.min_dots = min_dots
        self.hold_frames = hold_frames
        # 离开对局需要连续失败 3 倍进入帧数（滞回，防抖动）
        self.leave_frames = hold_frames * 3
        self.post_hold_s = post_hold_s   # 结算状态保持秒数，供下游感知"对局结束"

        self.phase = MatchPhase.WAITING
        self._in_match_streak = 0
        self._out_match_streak = 0
        self._post_since = None

    # ---------- 特征 ----------
    def _minimap_features(self, frame):
        h, w = frame.shape[:2]
        x1 = max(8, int(_MM_X_NORM * w))
        y1 = max(8, int(_MM_Y_NORM * h))
        roi = frame[0:y1, 0:x1]
        if roi.size == 0:
            return False, 0, 0
        gray = float(roi.mean())
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(int)
        S = hsv[..., 1].astype(int)
        V = hsv[..., 2].astype(int)
        green = int(np.sum((H >= 35) & (H <= 90) & (S > 80) & (V > 90)))
        B = roi[..., 0].astype(int)
        G = roi[..., 1].astype(int)
        R = roi[..., 2].astype(int)
        blue = int(np.sum((B >= 150) & (B <= 255) & (G >= 60) & (G <= 150)
                          & (R >= 30) & (R <= 120)))
        return gray < self.gray_thr, green, blue

    # ---------- 状态机 ----------
    def update(self, frame):
        """输入一帧 BGR 画面，返回当前 MatchPhase。"""
        if frame is None:
            return self.phase
        dark, green, blue = self._minimap_features(frame)
        # v2.13：绿色地图底 = 对局铁证；蓝色超上限（导航栏）= 局外界面
        in_match = dark and green > self.green_thr and blue < self.blue_max

        if in_match:
            # v3.0: POST_MATCH 状态锁定（结算界面绿色元素抖动不得拉回对局）
            if self.phase == MatchPhase.POST_MATCH:
                if self._post_since and time.time() - self._post_since > self.post_hold_s:
                    self.phase = MatchPhase.WAITING
                return self.phase
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
                if self._out_match_streak >= self.leave_frames:
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
