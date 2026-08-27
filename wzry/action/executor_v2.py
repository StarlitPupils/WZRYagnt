# -*- coding: utf-8 -*-
"""动作执行器 v0（M2）：把策略输出（连续角度/幅度/技能意图）翻译为触摸原语。

基于 wzry.control.executor.AdbExecutor，在【设备像素】坐标系工作（1280x720 横屏）。
坐标来源：configs/calibration_absolute.json（move_stick_center / dir_* / skill1-3 / attack）。

原语：
  - move(theta, r, duration)    摇杆任意方向（theta 弧度，r 0-1）+ 按住时长
  - move_dir(name)              8 向快捷移动（校准点直用）
  - skill_cast(skill_id, mode, target)  点按 / 拖到目标点瞄准 / 取消
  - attack(priority)            攻击键（自由/补兵/推塔）
  - sequence([...])             连招编排（每步原语 + 间隔）
"""
import json
import math
from pathlib import Path

from wzry.control.executor import AdbExecutor

ROOT = Path(__file__).resolve().parents[2]
CALIB = json.loads((ROOT / "configs" / "calibration_absolute.json").read_text(encoding="utf-8"))
PTS = CALIB["points"]


class ActionExecutor:
    def __init__(self, executor: AdbExecutor = None, stick_radius: float = 0.12):
        """stick_radius: 摇杆拖动半径（相对画面高度，默认 0.12 = 720p 下 ~86px）。"""
        self.ex = executor or AdbExecutor()
        self.stick_radius = stick_radius
        self.cx, self.cy = PTS["move_stick_center"]
        self.skills = {1: PTS["skill1"], 2: PTS["skill2"], 3: PTS["skill3"]}
        self.attack_pts = {"free": PTS["attack"], "minion": PTS["attack_minion"],
                           "tower": PTS["attack_tower"]}

    # ---------- 坐标换算 ----------
    def _stick_point(self, theta: float, r: float):
        """角度/幅度 -> 摇杆终点（设备像素）。theta 0=右，逆时针。"""
        r = max(0.0, min(1.0, r))
        h = CALIB["screen_height"]
        radius = self.stick_radius * h
        return (int(self.cx + radius * r * math.cos(theta)),
                int(self.cy - radius * r * math.sin(theta)))

    # ---------- 移动 ----------
    def move(self, theta: float, r: float = 1.0, duration: int = 300):
        """任意方向移动：按住摇杆 duration 毫秒。"""
        x, y = self._stick_point(theta, r)
        return self.ex.swipe(self.cx, self.cy, x, y, duration, source="policy")

    def move_dir(self, name: str, duration: int = 300):
        """8 向快捷移动（校准点）。name: up/left_up/left/..."""
        key = f"dir_{name}"
        if key not in PTS:
            raise ValueError(f"未知方向: {name}")
        x, y = PTS[key]
        return self.ex.swipe(self.cx, self.cy, x, y, duration, source="policy")

    def stop(self):
        """松开摇杆（同点 1ms swipe）。"""
        return self.ex.swipe(self.cx, self.cy, self.cx, self.cy, 1, source="policy")

    # ---------- 小地图点击移动（v2.27 王者辅助移动）----------
    def tap(self, x: int, y: int, source: str = "policy"):
        """点击设备像素坐标（小地图目标点→自动寻路）。"""
        return self.ex.tap(x, y, source=source)

    # ---------- 技能 ----------
    def skill_cast(self, skill_id: int, mode: str = "tap", target=None, duration: int = 150):
        """技能释放。
        mode='tap'      点按（智能施法）
        mode='drag'     从技能按钮拖到 target=(x,y) 设备像素（手动瞄准）
        mode='cancel'   按下后拖回按钮位置取消
        """
        if skill_id not in self.skills:
            raise ValueError(f"未知技能: {skill_id}")
        sx, sy = self.skills[skill_id]
        if mode == "tap":
            return self.ex.tap(sx, sy, source="policy")
        if mode == "drag":
            if target is None:
                raise ValueError("drag 模式需要 target=(x,y)")
            return self.ex.swipe(sx, sy, int(target[0]), int(target[1]), duration,
                                 source="policy")
        if mode == "cancel":
            return self.ex.swipe(sx, sy, sx, sy + 60, duration, source="policy")
        raise ValueError(f"未知模式: {mode}")

    def attack(self, priority: str = "free"):
        """攻击键切换。priority: free/minion/tower（分别点击对应按键）。"""
        if priority not in self.attack_pts:
            raise ValueError(f"未知攻击优先级: {priority}")
        x, y = self.attack_pts[priority]
        return self.ex.tap(x, y, source="policy")

    def summoner(self):
        """召唤师技能（校准点 summoner，默认闪现/惩击位）。"""
        x, y = PTS["summoner"]
        return self.ex.tap(x, y, source="policy")

    def recall(self):
        """回城（校准点 recall；点按触发回城读条，持续 7 秒）。"""
        x, y = PTS["recall"]
        return self.ex.tap(x, y, source="policy")

    def restore(self):
        """恢复键（校准点 restore；回复药/恢复，有冷却）。"""
        x, y = PTS["restore"]
        return self.ex.tap(x, y, source="policy")


if __name__ == "__main__":
    pass

    # ---------- 连招 ----------
    def sequence(self, steps, gap_ms: int = 120):
        """连招编排。steps: [(callable, args, kwargs), ...] 或原语 dict 列表。
        dict 格式: {"move": {"theta": 0, "r": 1, "duration": 200}} /
                   {"skill": {"id": 2, "mode": "tap"}} /
                   {"attack": {"priority": "free"}} / {"stop": {}}
        """
        import time
        for step in steps:
            if isinstance(step, dict):
                self._exec_dict(step)
            else:
                fn, args, kwargs = step
                fn(*args, **kwargs)
            time.sleep(gap_ms / 1000)

    def _exec_dict(self, d: dict):
        for name, kw in d.items():
            if name == "move":
                self.move(**kw)
            elif name == "move_dir":
                self.move_dir(**kw)
            elif name == "stop":
                self.stop()
            elif name == "skill":
                kw = dict(kw)
                kw["skill_id"] = kw.pop("id")
                self.skill_cast(**kw)
            elif name == "attack":
                self.attack(**kw)
            else:
                raise ValueError(f"未知原语: {name}")
