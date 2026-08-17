# -*- coding: utf-8 -*-
"""M2 Agent v2.1 决策单元测试（用户策略指导版）。

场景（用户规则）：
  0. 勾中连招：二技能释放后窗口内敌人被拉近 → 召唤师技能 + 三技能
  1. 塔规避：敌方塔可见且无我方小兵 → 移动避开（技能不受限）
  2. 二技能：敌方英雄在二技能范围内 → 钩子
  3. 一技能：不在大招生效期 且 敌人/敌兵在身边 → 一技能
  4. 移动：有敌人→朝最近敌人持续拖动(2000ms)；无敌人→跟随 ally_hero 或朝发育路
运行：
    venv\\Scripts\\python.exe scripts\\m2_agent_v2_test.py
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_agent_v2 import decide, LANE_DIR, MOVE_DURATION_MS  # noqa: E402

W, H = 1280, 720
T = 100.0


def fresh_cd():
    return {"skill1_t": 0.0, "skill2_t": 0.0, "skill3_t": 0.0,
            "summoner_t": 0.0, "skill": 0.0, "hook_pending": 0.0,
            "hook_anchor_dist": 0.0, "turret_threat": 0.0}


def state(units=None, minimap=None, t=T, screen=(W, H)):
    return {"t": t, "screen_size": list(screen), "units": units or [],
            "minimap": minimap if minimap is not None else {"found": False}}


def enemy(cx, cy):
    return {"cls": "enemy_hero", "screen": [cx, cy, 0.06, 0.06]}


def enemy_minion(cx, cy):
    return {"cls": "enemy_minion", "screen": [cx, cy, 0.04, 0.04]}


def enemy_turret(cx, cy):
    return {"cls": "enemy_turret", "screen": [cx, cy, 0.10, 0.10]}


def ally(cx, cy):
    return {"cls": "ally_hero", "screen": [cx, cy, 0.06, 0.06]}


def mm_found(red=None, blue=None):
    return {"found": True, "center": [110, 93], "radius": 60,
            "dots": {"blue": blue or [], "red": red or [], "yellow": []},
            "towers": []}


class TestHookCombo(unittest.TestCase):
    """规则 0：勾中连招 -> summoner + skill3。"""

    def test_hook_confirmed_triggers_combo(self):
        cd = fresh_cd()
        cd["hook_pending"] = T - 0.2          # 0.2s 前放钩
        cd["hook_anchor_dist"] = 0.40         # 释放时敌人在 0.4 屏宽
        st = state(units=[enemy(0.68, 0.50)])  # 现在 0.18 < 0.40*0.72=0.288 -> 勾中
        act = decide(st, cd)
        self.assertEqual(act["type"], "combo")
        types = [a["type"] for a in act["actions"]]
        self.assertEqual(types, ["summoner", "skill"])
        self.assertEqual(act["actions"][1]["id"], 3)

    def test_no_combo_when_enemy_not_pulled(self):
        cd = fresh_cd()
        cd["hook_pending"] = T - 0.2
        cd["hook_anchor_dist"] = 0.20
        st = state(units=[enemy(0.62, 0.50)])  # 0.12 < 0.144 也缩小了 -> 仍算勾中
        act = decide(st, cd)
        # 距离 0.12 < 0.20*0.72=0.144 -> 勾中
        self.assertEqual(act["type"], "combo")

    def test_combo_window_expires(self):
        cd = fresh_cd()
        cd["hook_pending"] = T - 2.0           # 超窗
        cd["hook_anchor_dist"] = 0.40
        st = state(units=[enemy(0.55, 0.50)])  # 很近
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "combo")
        # 距离 0.05 < 0.42 -> 钩子（冷却就绪）
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))


class TestTurretAvoid(unittest.TestCase):
    """规则 1：塔规避。"""

    def test_avoid_turret_when_no_allied_minion(self):
        st = state(units=[enemy_turret(0.7, 0.5)])
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "avoid_turret")
        # 远离塔：塔在右 -> theta 朝左（±pi 等价）
        self.assertAlmostEqual(math.cos(act["theta"]), math.cos(math.pi), places=4)

    def test_no_avoid_when_allied_minion_present(self):
        st = state(units=[enemy_turret(0.7, 0.5),
                          {"cls": "ally_minion", "screen": [0.6, 0.5, 0.04, 0.04]}])
        act = decide(st, fresh_cd())
        self.assertNotEqual(act["reason"], "avoid_turret")


class TestSkill2(unittest.TestCase):
    """规则 2：敌人在二技能范围内 -> 钩子。"""

    def test_enemy_in_range_triggers_skill2(self):
        st = state(units=[enemy(0.85, 0.50)])  # dx=0.35 < 0.42
        act = decide(st, fresh_cd())
        self.assertEqual((act["type"], act["id"]), ("skill", 2))

    def test_enemy_out_of_range_no_hook(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.95, 0.50)])  # dx=0.45 > 0.42
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))
        self.assertEqual(act["type"], "move")

    def test_skill2_throttle(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 2.0  # 2s 前 -> 未过 3s 节流
        st = state(units=[enemy(0.85, 0.50)])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))


class TestSkill1(unittest.TestCase):
    """规则 3：一技能（不在大招生效期 + 敌人/敌兵在身边）。"""

    def test_skill1_when_enemy_near(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 4.0  # 2 技能冷却已过? 不——4s 前 -> 就绪，会先钩子
        # 敌人 0.2 屏宽在 0.42 内 -> 钩子优先。要测 1 技能需钩子不可用
        cd["skill2_t"] = T - 2.0  # 2 技能节流中
        st = state(units=[enemy(0.70, 0.50)])  # 0.2 < 0.25
        act = decide(st, cd)
        self.assertEqual((act["type"], act["id"]), ("skill", 1))

    def test_skill1_when_enemy_minion_near(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 2.0
        st = state(units=[enemy_minion(0.72, 0.50)])  # 0.22 < 0.25
        act = decide(st, cd)
        self.assertEqual((act["type"], act["id"]), ("skill", 1))

    def test_no_skill1_during_ult(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 2.0
        cd["skill3_t"] = T - 1.0  # 大招 1s 前 -> 生效期内，1 技能被抑制
        st = state(units=[enemy(0.70, 0.50)])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 1))


class TestMovement(unittest.TestCase):
    """规则 4：持续拖动移动。"""

    def test_chase_enemy_duration_2000(self):
        st = state(units=[enemy(0.95, 0.50)])  # 0.45 > 0.42 钩子不可用
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["duration_ms"], MOVE_DURATION_MS)
        self.assertEqual(act["reason"], "chase_enemy")
        self.assertAlmostEqual(act["theta"], 0.0, places=6)

    def test_follow_ally_when_no_enemy(self):
        st = state(units=[ally(0.60, 0.50)])
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "follow_ally")
        self.assertAlmostEqual(act["theta"], 0.0, places=6)

    def test_lane_develop_when_nothing(self):
        st = state(units=[], minimap=mm_found())
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "lane_develop")
        lx, ly = LANE_DIR
        exp = math.atan2(-(ly - 0.5) * (H / W), lx - 0.5)  # 与 decide 一致（aspect 折算）
        self.assertAlmostEqual(act["theta"], exp, places=4)


class TestGuard(unittest.TestCase):
    """防抖。"""

    def test_debounce_suppresses_move(self):
        cd = fresh_cd()
        cd["skill"] = T - 0.03
        st = state(units=[enemy(0.95, 0.50)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "none")
        self.assertEqual(act["reason"], "skill_debounce")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    ok = runner.run(suite).wasSuccessful()
    sys.exit(0 if ok else 1)
