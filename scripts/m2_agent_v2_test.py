# -*- coding: utf-8 -*-
"""M2 Agent v2 决策单元测试：用 mock 的"检测结果 + 小地图"输入测试 decide()，
断言输出动作类型与参数；不依赖真机 / 模型 / 触摸。

构造 4 个核心场景：
  1. 钩子命中（hook_aim 或 敌人在 2 技能距离内） -> skill 2
  2. 敌人近（中心距屏幕中心 < 0.25 屏宽）         -> skill 1
  3. 敌人远                                       -> move(朝向敌人)
  4. 无敌人                                       -> 朝小地图红点质心移动 / idle
外加冷却节流与技能后 50ms 防抖用例。

运行：
    venv\\Scripts\\python.exe scripts\\m2_agent_v2_test.py
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_agent_v2 import decide  # noqa: E402

W, H = 1280, 720
T = 100.0  # 统一的"现在"时刻，冷却时间戳相对它构造


def fresh_cd():
    """全部技能从未释放 -> 均就绪。"""
    return {"skill1": 0.0, "skill2": 0.0, "skill": 0.0}


def state(units=None, minimap=None, t=T, screen=(W, H)):
    return {"t": t, "screen_size": list(screen), "units": units or [],
            "minimap": minimap if minimap is not None else {"found": False}}


def enemy(cx, cy):
    return {"cls": "enemy_hero", "screen": [cx, cy, 0.06, 0.06]}


def hook():
    return {"cls": "hook_aim", "screen": [0.5, 0.5, 0.10, 0.10]}


def mm_found(red=None, blue=None):
    return {"found": True, "center": [110, 93], "radius": 60,
            "dots": {"blue": blue or [], "red": red or [], "yellow": []},
            "towers": []}


class TestHookCanHit(unittest.TestCase):
    """场景 1：钩子可命中 -> skill 2。"""

    def test_hook_aim_detected_triggers_skill2(self):
        # hook_aim 指示线出现（敌人即使很远也按钩子优先）
        st = state(units=[enemy(0.90, 0.50), hook()])
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 2)
        self.assertEqual(act["mode"], "tap")
        self.assertEqual(act["reason"], "hook_aim")

    def test_enemy_in_skill2_range_triggers_skill2(self):
        # 无 hook_aim，但敌人进入 2 技能距离（0.35 屏宽 < 0.42）
        st = state(units=[enemy(0.85, 0.50)])  # dx=0.35
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 2)
        self.assertEqual(act["reason"], "enemy_in_skill2_range")

    def test_enemy_near_still_hook_priority(self):
        # 敌人很近（0.2 屏宽）且 2 技能就绪：按优先级仍先出钩子
        st = state(units=[enemy(0.70, 0.50)])  # dx=0.20 < 0.25 也 < 0.42
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 2)


class TestEnemyNear(unittest.TestCase):
    """场景 2：敌人近 -> skill 1（2 技能冷却时回退到 1 技能）。"""

    def test_enemy_near_triggers_skill1_when_skill2_cd(self):
        # 敌人 0.2 屏宽：在 2 技能距离内，但 2 技能 1s 前刚放（>3s 节流未满足）
        cd = fresh_cd()
        cd["skill2"] = T - 1.0
        st = state(units=[enemy(0.70, 0.50)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 1)
        self.assertEqual(act["reason"], "enemy_near")

    def test_enemy_near_throttled_skill1_falls_to_move(self):
        # 敌人近，但 1 技能 1s 前刚放（>1.5s 节流未满足），2 技能也在冷却
        cd = fresh_cd()
        cd["skill2"] = T - 1.0
        cd["skill1"] = T - 1.0
        st = state(units=[enemy(0.70, 0.50)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertAlmostEqual(act["theta"], 0.0, places=6)  # 敌人正右方


class TestEnemyFar(unittest.TestCase):
    """场景 3：敌人远 -> move(朝向最近敌人, r=0.8, 400ms)。"""

    def test_enemy_far_moves_toward_enemy(self):
        # 敌人 0.45 屏宽（> 0.42 不在 2 技能距离，> 0.25 不近）
        st = state(units=[enemy(0.95, 0.50)])
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["r"], 0.8)
        self.assertEqual(act["duration_ms"], 400)
        self.assertAlmostEqual(act["theta"], 0.0, places=6)
        self.assertEqual(act["reason"], "chase_enemy")

    def test_move_theta_points_at_below_right_enemy(self):
        # 敌人位于屏幕中心右下且超出 2 技能距离（d≈0.46 屏宽）：
        # theta 应为负（向下），幅值 atan2(-dy*aspect, dx)
        st = state(units=[enemy(0.95, 0.70)])
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        exp = math.atan2(-(0.70 - 0.5) * (H / W), 0.95 - 0.5)
        self.assertAlmostEqual(act["theta"], exp, places=6)

    def test_chase_picks_nearest_enemy(self):
        # 多个敌人：选最近的那个。两个都超出近身阈值（>0.25 屏宽），
        # 且 2 技能在冷却（避免钩子分支），验证 theta 指向较近者 (0.75, 0.30)
        cd = fresh_cd()
        cd["skill2"] = T - 1.0  # 2 技能冷却，避免钩子分支
        st = state(units=[enemy(0.95, 0.50), enemy(0.75, 0.30)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        exp = math.atan2(-(0.30 - 0.5) * (H / W), 0.75 - 0.5)  # 最近者 (0.75,0.30)
        self.assertAlmostEqual(act["theta"], exp, places=6)


class TestNoEnemy(unittest.TestCase):
    """场景 4：无敌人 -> 朝小地图红点质心移动；无红点则 idle。"""

    def test_moves_to_red_centroid(self):
        # 红点质心在 (0.7, 0.5)（正右）-> theta ≈ 0
        st = state(units=[], minimap=mm_found(red=[[0.7, 0.5]]))
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["r"], 0.8)
        self.assertEqual(act["duration_ms"], 400)
        self.assertAlmostEqual(act["theta"], 0.0, places=6)
        self.assertEqual(act["reason"], "lane_red_centroid")

    def test_red_centroid_theta_down(self):
        # 质心 (0.5, 0.8)（正下，小地图 ny 向下）-> theta ≈ -pi/2
        st = state(units=[], minimap=mm_found(red=[[0.5, 0.8], [0.5, 0.6]]))
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertAlmostEqual(act["theta"], -math.pi / 2, places=6)

    def test_red_centroid_averages_dots(self):
        # 两个红点 (0.6,0.4)+(0.6,0.6) -> 质心 (0.6, 0.5) -> theta ≈ 0
        st = state(units=[], minimap=mm_found(red=[[0.6, 0.4], [0.6, 0.6]]))
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertAlmostEqual(act["theta"], 0.0, places=6)

    def test_no_red_dots_idle(self):
        st = state(units=[], minimap=mm_found(red=[]))
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "none")

    def test_no_minimap_idle(self):
        st = state(units=[], minimap={"found": False})
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "none")

    def test_empty_state_idle(self):
        act = decide(state(), fresh_cd())
        self.assertEqual(act["type"], "none")


class TestCooldownAndDebounce(unittest.TestCase):
    """冷却节流 + 技能后 50ms 防抖。"""

    def test_skill2_throttle_over_3s(self):
        cd = fresh_cd()
        cd["skill2"] = T - 3.1  # 3.1s 前 -> 刚过 3s 节流 -> 可再放
        st = state(units=[enemy(0.90, 0.50), hook()])
        act = decide(st, cd)
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 2)

    def test_skill2_throttle_blocks(self):
        cd = fresh_cd()
        cd["skill2"] = T - 2.0  # 2s 前 -> 未过 3s 节流 -> 不放大招级钩子
        st = state(units=[enemy(0.95, 0.50), hook()])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))

    def test_skill1_throttle_blocks(self):
        cd = fresh_cd()
        cd["skill2"] = T - 1.0  # 2 技能冷却，逼到 1 技能分支
        cd["skill1"] = T - 1.0  # 1 技能 1s 前 -> 未过 1.5s 节流
        st = state(units=[enemy(0.70, 0.50)])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 1))
        self.assertEqual(act["type"], "move")  # 回退为移动

    def test_skill_debounce_suppresses_move(self):
        cd = fresh_cd()
        cd["skill"] = T - 0.03  # 30ms 前刚放技能 -> 移动被防抖
        st = state(units=[enemy(0.95, 0.50)])  # 本应 chase_enemy
        act = decide(st, cd)
        self.assertEqual(act["type"], "none")
        self.assertEqual(act["reason"], "skill_debounce")

    def test_skill_debounce_blocks_lane_move(self):
        cd = fresh_cd()
        cd["skill"] = T - 0.01
        st = state(units=[], minimap=mm_found(red=[[0.7, 0.5]]))
        act = decide(st, cd)
        self.assertEqual(act["type"], "none")
        self.assertEqual(act["reason"], "skill_debounce")

    def test_skill_debounce_does_not_block_skill(self):
        # 防抖只针对移动；技能本身仍可放（钩子优先）
        cd = fresh_cd()
        cd["skill"] = T - 0.03
        st = state(units=[enemy(0.70, 0.50)])  # 近且 2 技能就绪
        act = decide(st, cd)
        self.assertEqual(act["type"], "skill")
        self.assertEqual(act["id"], 2)

    def test_no_action_when_all_blocked(self):
        cd = fresh_cd()
        cd["skill"] = T - 0.03  # 防抖
        st = state(units=[])    # 且无红点
        act = decide(st, cd)
        self.assertEqual(act["type"], "none")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    ok = runner.run(suite).wasSuccessful()
    sys.exit(0 if ok else 1)
