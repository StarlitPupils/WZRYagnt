# -*- coding: utf-8 -*-
"""理解层（v2.9）：把屏幕像素 → 结构化局势理解（供决策层消费）。

信息源（用户语义确认 + 真机标定）：
  1. 小地图（方形 1280x720 下 (0,0)-(232,232)，中心 116,116）：
     - 绿色大圈+箭头 = 自己英雄（H 20-90, S>50, V>50）
     - 蓝色大圈 = 队友英雄（H 90-140, S>80, V>80）
     - 红色 = 敌方英雄（H<=15 或 H>=165, S>100, V>90）
     - 黄色小点 = 野怪
     - 蓝/红小点 = 我方/敌方小兵（小尺寸）
     - 蓝/红方块 = 我方/敌方塔
  2. 自己技能状态：技能按钮 mean_v/dark_frac（已实现 skill_ready_state）
  3. 自己血量/蓝量：HP 条检测（find_hp_bar）+ MP 条（find_mp_bar）
  4. 队友头像区（顶部中间）：头像+血条+技能图标（待完善）
  5. 敌方信息（右上角）：头像+名字+血量（待完善）

用法（对局中每帧调用）：
    from wzry.vision.understanding import Understanding
    u = Understanding()
    status = u.update(frame)   # -> 结构化 dict（见 update 返回）
"""
from __future__ import annotations

import cv2
import numpy as np

from wzry.vision import minimap
from wzry.vision.terrain import DEFAULT_BOX


class Understanding:
    """理解层：帧 → 结构化局势。"""

    def __init__(self, mm_box=None):
        # 小地图方形框（1280x720 标定，等比缩放）
        self.mm_box = list(mm_box) if mm_box else list(DEFAULT_BOX)
        self._last = None

    # ---------- 小地图 ----------
    def _minimap(self, frame):
        """解析小地图：绿=自己、蓝=队友、红=敌、黄=野怪、塔、小兵。"""
        x0, y0, x1, y1 = self.mm_box
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = (x1 - x0) // 2
        fake = {"found": True, "center": [cx, cy], "radius": r}
        det = minimap.detect_dots(frame, fake)
        return {
            "center": [cx, cy], "radius": r,
            "self": det["dots"].get("green", []),        # 自己（绿圈）
            "allies": det["dots"].get("blue", []),       # 队友（蓝圈）
            "enemies": det["dots"].get("red", []),       # 敌人（红圈）
            "monsters": det["dots"].get("yellow", []),   # 野怪（黄点）
            "ally_minions": det["minions"].get("blue", []) + det["minions"].get("green", []),
            "enemy_minions": det["minions"].get("red", []),
            "towers": det["towers"],                     # 塔（含 color 字段）
        }

    # ---------- 自己状态 ----------
    def _self_status(self, frame):
        """自己：HP/MP/技能状态。

        HP/MP 用英雄头顶血条蓝条检测（self_bars，镜头跟随自己）。
        """
        from wzry.vision.ui_reader import skill_ready_state
        skills_pts = _load_skill_points()
        skill_states = skill_ready_state(frame, skills_pts)
        hp = mp = None
        try:
            from wzry.vision.self_bars import detect_self_bars
            hp, mp, hero_pos = detect_self_bars(frame)
        except Exception:
            hero_pos = None
        # 回退：HUD 血条检测
        if hp is None:
            try:
                from wzry.vision.ui_reader import find_hp_bar
                hb = find_hp_bar(frame)
                if hb:
                    hp = hb[-1]
            except Exception:
                pass
        return {
            "hp": hp,
            "mp": mp,
            "hero_pos": hero_pos,
            "skills": skill_states,
        }

    # ---------- 队友/敌人（顶部头像条 + 血条颜色定位） ----------
    def _others(self, frame):
        """队友头像条 + 三色血条（绿=自己/蓝=队友/红=敌人）位置。"""
        result = {"teammates": [], "enemies": [], "bar_self": [],
                  "bar_allies": [], "bar_enemies": []}
        try:
            from wzry.vision.teammate_bar import detect_teammates, detect_enemies
            result["teammates"] = detect_teammates(frame)
            result["enemies"] = detect_enemies(frame)
        except Exception:
            pass
        try:
            from wzry.vision.self_bars import detect_all_bars
            bars = detect_all_bars(frame)
            result["bar_self"] = bars.get("self", [])
            result["bar_allies"] = bars.get("allies", [])
            result["bar_enemies"] = bars.get("enemies", [])
        except Exception:
            pass
        return result

    # ---------- 英雄识别（小地图英雄圈 -> 英雄名） ----------
    def _recognize_minimap_heroes(self, frame):
        """识别小地图上每个英雄圈对应的英雄名（裁剪圈内头像 -> 模板匹配）。"""
        mm = self._last.get("minimap") if self._last else None
        if mm is None:
            return {"self": None, "allies": [], "enemies": []}
        try:
            from wzry.vision.hero_recognition import HeroRecognizer
            rec = HeroRecognizer()
        except Exception:
            rec = None
        x0, y0, x1, y1 = self.mm_box
        out = {"self": None, "allies": [], "enemies": []}
        if rec is None:
            return out

        def identify(pos):
            """裁剪小地图英雄圈内头像并识别。pos=(nx,ny) 归一化。"""
            nx, ny = pos
            cx = int(x0 + nx * (x1 - x0))
            cy = int(y0 + ny * (y1 - y0))
            r = int((x1 - x0) * 0.06)   # 英雄圈半径（小地图宽度的6%）
            crop = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            res = rec.recognize(crop)
            return res[0][0] if res else None

        if mm.get("self"):
            out["self"] = {"pos": mm["self"][0], "hero": identify(mm["self"][0])}
        for p in mm.get("allies", []):
            out["allies"].append({"pos": p, "hero": identify(p)})
        for p in mm.get("enemies", []):
            out["enemies"].append({"pos": p, "hero": identify(p)})
        return out

    # ---------- 组合入口 ----------
    def update(self, frame, recognize=False):
        """帧 → 结构化理解 dict。recognize=True 时额外做英雄识别（较慢）。"""
        status = {
            "minimap": self._minimap(frame),
            "self": self._self_status(frame),
            "others": self._others(frame),
        }
        if recognize:
            status["heroes"] = self._recognize_minimap_heroes(frame)
        self._last = status
        return status

    @property
    def last(self):
        return self._last


def _load_skill_points():
    """加载技能按钮校准坐标（供 ui_reader 使用）。"""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "configs" / "calibration_absolute.json"
    pts = json.loads(p.read_text(encoding="utf-8"))["points"]
    return {1: pts["skill1"], 2: pts["skill2"], 3: pts["skill3"]}
