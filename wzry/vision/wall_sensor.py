# -*- coding: utf-8 -*-
"""撞墙感知（防卡墙）：检测"轮盘在拖但英雄没移动"= 撞墙，自动绕行。

原理（用户规则：移动时 W/D 穿插防卡墙）：
  - 位置信号：小地图上的己方英雄点（蓝点，离圆心最近的那个通常是己方）
    ——小地图是全局坐标，英雄点不动 = 英雄真没动（比画面中心可靠）。
  - 撞墙判定：轮盘持续拖动（move 指令）时，蓝点位置在窗口内几乎不变
    （位移 < 阈值）且持续超过 N 帧 -> 判定撞墙。
  - 绕行：切换移动方向为垂直偏移（左右随机/交替），持续短时间后再回到目标方向。
  - 恢复：蓝点恢复移动后回到正常寻路。

用法（集成进 Agent 主循环）：
    from wzry.vision.terrain import WallSensor
    ws = WallSensor()
    ...
    wall_hit = ws.update(now, is_moving=True, hero_pos=mm_blue_self)
    if wall_hit:  # 撞墙 -> 绕行指令
        action = ws.avoid_action()
"""
from __future__ import annotations

import math
import time


class WallSensor:
    """撞墙检测器（基于小地图己方蓝点位置）。"""

    def __init__(self, still_px=0.012, frames=6, avoid_ms=600):
        """
        still_px : 蓝点移动阈值（小地图归一化坐标位移，< 该值视为没动）
        frames   : 连续"没动"帧数达到该值判定撞墙
        avoid_ms : 绕行持续时间（毫秒）
        """
        self.still_px = still_px
        self.frames = frames
        self.avoid_ms = avoid_ms
        self._still_streak = 0
        self._last_pos = None
        self._last_t = 0.0
        self._avoiding = False
        self._avoid_until = 0.0
        self._avoid_dir = 1.0       # 左右交替
        self._avoid_count = 0

    def update(self, now: float, is_moving: bool, hero_pos):
        """每感知帧调用。

        now      : wall clock 秒
        is_moving: 本帧是否在持续移动（轮盘拖动中）
        hero_pos : 己方英雄在小地图的归一化坐标 (nx, ny) 或 None（未定位）

        返回 True 表示刚判定撞墙（本帧应改为绕行）。
        """
        # 绕行期间：结束后复位
        if self._avoiding:
            if now >= self._avoid_until:
                self._avoiding = False
                self._still_streak = 0
            return False
        if not is_moving or hero_pos is None:
            self._still_streak = 0
            self._last_pos = hero_pos
            return False
        if self._last_pos is None:
            self._last_pos = hero_pos
            return False
        dx = hero_pos[0] - self._last_pos[0]
        dy = hero_pos[1] - self._last_pos[1]
        moved = math.hypot(dx, dy)
        self._last_pos = hero_pos
        if moved < self.still_px:
            self._still_streak += 1
            if self._still_streak >= self.frames:
                self._still_streak = 0
                self._avoiding = True
                self._avoid_until = now + self.avoid_ms / 1000.0
                self._avoid_dir = -self._avoid_dir if self._avoid_count > 0 else 1.0
                self._avoid_count += 1
                return True
        else:
            self._still_streak = max(0, self._still_streak - 1)
        return False

    def avoid_action(self, target_theta: float):
        """撞墙后的绕行指令：目标方向垂直偏移（左右交替）。"""
        theta = target_theta + self._avoid_dir * (math.pi / 2)
        return {"type": "move", "theta": theta, "r": 1.0,
                "duration_ms": self.avoid_ms, "reason": "wall_avoid"}

    @property
    def avoiding(self):
        return self._avoiding
