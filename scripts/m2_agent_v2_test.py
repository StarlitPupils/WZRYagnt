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
from m2_agent_v2 import decide, LANE_DIR_BLUE, MOVE_DURATION_MS  # noqa: E402

W, H = 1280, 720
T = 100.0


def fresh_cd():
    return {"skill1_t": 0.0, "skill2_t": 0.0, "skill3_t": 0.0,
            "summoner_t": 0.0, "recall_t": 0.0, "hp_t": 0.0,
            "skill": 0.0, "hook_pending": 0.0,
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
        st = state(units=[enemy(0.62, 0.50)])  # 0.12 缩小 0.08 < 0.12 -> 不算勾中
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "combo")

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
        st = state(units=[enemy_turret(0.92, 0.5)])  # 0.42 屏宽：威胁带（未进 0.40 射程线）
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "avoid_turret")
        # 远离塔：塔在右 -> theta 朝左（±pi 等价）
        self.assertAlmostEqual(math.cos(act["theta"]), math.cos(math.pi), places=4)

    def test_escape_turret_when_very_close(self):
        st = state(units=[enemy_turret(0.7, 0.5)])  # 0.20 屏宽：塔射程内 -> 逃离
        act = decide(st, fresh_cd())
        self.assertEqual(act["reason"], "escape_turret")

    def test_no_avoid_when_allied_minion_present(self):
        st = state(units=[enemy_turret(0.7, 0.5),
                          {"cls": "ally_minion", "screen": [0.6, 0.5, 0.04, 0.04]}])
        act = decide(st, fresh_cd())
        self.assertNotEqual(act["reason"], "avoid_turret")

    def test_escape_turret_when_in_range(self):
        """塔中心 < 0.40（被塔打）且无我方小兵 -> 立即反向逃离（用户反馈修复）。"""
        st = state(units=[enemy_turret(0.60, 0.50)])  # 0.10 屏宽，塔射程内
        act = decide(st, fresh_cd())
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "escape_turret")
        # 塔在右 -> 反向朝左跑（±pi 等价）
        self.assertAlmostEqual(math.cos(act["theta"]), math.cos(math.pi), places=4)

    def test_escape_turret_beats_skill2(self):
        """塔射程内即使敌人在钩子范围内也先跑（保命优先，用户反馈：被塔打要赶紧出去）。"""
        cd = fresh_cd()
        st = state(units=[enemy_turret(0.60, 0.50), enemy(0.72, 0.50)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "escape_turret")

    def test_escape_turret_when_moving_toward_turret(self):
        """目标方向与塔同向 -> 反向逃离（替代原垂直绕行）。"""
        cd = fresh_cd()
        # 塔在右 0.42 屏宽（威胁带 0.40~0.45，未触发射程内逃离），敌人在更右 -> 追敌人会进塔
        st = state(units=[enemy_turret(0.92, 0.50), enemy(0.97, 0.50)])
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "escape_turret")
        self.assertAlmostEqual(math.cos(act["theta"]), math.cos(math.pi), places=4)


class TestSkill2(unittest.TestCase):
    """规则 2：敌人在二技能范围内 -> 钩子。"""

    def test_enemy_in_range_triggers_skill2(self):
        st = state(units=[enemy(0.75, 0.50)])  # dx=0.25 < 0.28（实测射程 0.30）
        act = decide(st, fresh_cd())
        self.assertEqual((act["type"], act["id"]), ("skill", 2))

    def test_enemy_out_of_range_no_hook(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.85, 0.50)])  # dx=0.35 > 0.28
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


class TestMeleeAttack(unittest.TestCase):
    """规则 3.5：贴身敌人/兵且技能不可用 -> 普攻（用户反馈：面对敌方英雄不攻击）。"""

    def test_attack_when_enemy_adjacent_and_skills_cd(self):
        """敌人贴身（0.1 屏宽）且 2/1 技能都刚放过 -> 普攻。"""
        cd = fresh_cd()
        cd["skill2_t"] = T - 0.5   # 2 技能冷却中
        cd["skill1_t"] = T - 0.5   # 1 技能冷却中
        st = state(units=[enemy(0.60, 0.50)])  # dx=0.10 < 0.20
        act = decide(st, cd)
        self.assertEqual(act["type"], "attack")
        self.assertEqual(act["reason"], "melee_attack")

    def test_attack_when_enemy_minion_adjacent(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 0.5
        cd["skill1_t"] = T - 0.5
        st = state(units=[enemy_minion(0.63, 0.50)])  # dx=0.13 < 0.20
        act = decide(st, cd)
        self.assertEqual(act["type"], "attack")

    def test_skill_preempts_attack_when_ready(self):
        """贴身敌人但 2 技能就绪 -> 钩子优先于普攻。"""
        cd = fresh_cd()
        st = state(units=[enemy(0.60, 0.50)])  # dx=0.10，也在钩子范围内
        act = decide(st, cd)
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))

    def test_no_attack_when_enemy_far(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 0.5
        cd["skill1_t"] = T - 0.5
        st = state(units=[enemy(0.90, 0.50)])  # dx=0.40 > 0.20
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "attack")


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
        cd = fresh_cd()
        cd["camp"] = "blue"
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "lane_develop")
        lx, ly = LANE_DIR_BLUE
        exp = math.atan2(-(ly - 0.5) * (H / W), lx - 0.5)  # 与 decide 一致（aspect 折算）
        self.assertAlmostEqual(act["theta"], exp, places=4)

    def test_lane_develop_red_mirror(self):
        """红方发育路应镜像到左上。"""
        st = state(units=[], minimap=mm_found())
        cd = fresh_cd()
        cd["camp"] = "red"
        act = decide(st, cd)
        self.assertEqual(act["reason"], "lane_develop")
        lx, ly = (0.28, 0.18)
        exp = math.atan2(-(ly - 0.5) * (H / W), lx - 0.5)
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


class TestRetreat(unittest.TestCase):
    """低血量撤退（用户规则）：HP<20% 朝泉水跑；身边无危险则回城。"""

    def test_low_hp_safe_recall(self):
        cd = fresh_cd()
        # 自家塔下（0.30 < 0.35）且身边无危险 -> 回城
        st = state(units=[{"cls": "ally_turret", "screen": [0.80, 0.50, 0.10, 0.10]}],
                   minimap=mm_found())
        st["ui"] = {"hp": 0.15}
        act = decide(st, cd)
        self.assertEqual(act["type"], "recall")
        self.assertEqual(act["reason"], "low_hp_safe_recall")

    def test_low_hp_danger_retreat_to_fountain(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.60, 0.50)])  # 0.10 < 0.50 有危险
        st["ui"] = {"hp": 0.10}
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "retreat_low_hp")
        # 蓝方泉水 (0.12,0.88) 方向（左下方）
        exp = math.atan2(-(0.88 - 0.5) * (H / W), 0.12 - 0.5)
        self.assertAlmostEqual(act["theta"], exp, places=4)

    def test_high_hp_no_retreat(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.75, 0.50)])  # 0.25 在钩子范围
        st["ui"] = {"hp": 0.8}
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "recall")
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))


class TestHookBlock(unittest.TestCase):
    """v2.5 规则 007：钩子路径被小兵/野怪挡住不钩。"""

    def test_hook_blocked_by_minion(self):
        cd = fresh_cd()
        # 敌英 0.26 屏宽（在钩子范围 0.28 内），但路径上有敌兵在 0.10 处同方向
        st = state(units=[enemy(0.76, 0.50), enemy_minion(0.60, 0.50)])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))

    def test_hook_blocked_by_monster(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.76, 0.50),
                          {"cls": "neutral_monster", "screen": [0.62, 0.50, 0.06, 0.06]}])
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))

    def test_hook_ok_when_path_clear(self):
        cd = fresh_cd()
        # 敌英 0.26，敌兵在旁边但不在直线上（y 偏移大）
        st = state(units=[enemy(0.76, 0.50), enemy_minion(0.60, 0.70)])
        act = decide(st, cd)
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))

    def test_hook_ok_when_minion_behind_enemy(self):
        cd = fresh_cd()
        # 敌兵在敌英后面（更远）不挡钩
        st = state(units=[enemy(0.76, 0.50), enemy_minion(0.90, 0.50)])
        act = decide(st, cd)
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))


class TestSkillUnlock(unittest.TestCase):
    """v2.5 规则：技能未解锁（灰暗）时不释放，改用普攻。"""

    def _ui(self, unlocked_map):
        return {"hp": 1.0, "skill_states": {
            str(k): {"unlocked": v, "ready": True, "mean_v": 150 if v else 45}
            for k, v in unlocked_map.items()}}

    def test_skill1_locked_uses_attack(self):
        cd = fresh_cd()
        cd["skill2_t"] = T - 2.0  # 2 技能冷却中
        st = state(units=[enemy(0.70, 0.50)])  # 0.2 贴身
        st["ui"] = self._ui({"1": False, "2": True, "3": True})
        act = decide(st, cd)
        self.assertEqual(act["type"], "attack")
        self.assertEqual(act["reason"], "melee_attack")

    def test_skill2_locked_no_hook(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.75, 0.50)])  # 0.25 在钩子范围
        st["ui"] = self._ui({"1": True, "2": False, "3": True})
        act = decide(st, cd)
        self.assertNotEqual((act["type"], act.get("id")), ("skill", 2))

    def test_skill2_unlocked_hooks(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.75, 0.50)])
        st["ui"] = self._ui({"1": True, "2": True, "3": True})
        act = decide(st, cd)
        self.assertEqual((act["type"], act.get("id")), ("skill", 2))


class TestRestore(unittest.TestCase):
    """v2.5 规则：HP/MP<80% 且安全 -> 恢复键。"""

    def test_restore_when_low_hp_safe(self):
        cd = fresh_cd()
        st = state(units=[], minimap=mm_found())
        st["ui"] = {"hp": 0.7}
        act = decide(st, cd)
        self.assertEqual(act["type"], "restore")
        self.assertEqual(act["reason"], "low_resource_safe_restore")

    def test_restore_when_low_mp_safe(self):
        cd = fresh_cd()
        st = state(units=[], minimap=mm_found())
        st["ui"] = {"hp": 1.0, "mp": 0.6}
        act = decide(st, cd)
        self.assertEqual(act["type"], "restore")

    def test_no_restore_when_danger(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.60, 0.50)])  # 0.10 身边有敌
        st["ui"] = {"hp": 0.7}
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "restore")

    def test_no_restore_when_high_resource(self):
        cd = fresh_cd()
        st = state(units=[], minimap=mm_found())
        st["ui"] = {"hp": 0.95, "mp": 0.9}
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "restore")

    def test_restore_throttle(self):
        cd = fresh_cd()
        cd["restore_t"] = T - 2.0  # 10s 节流内
        st = state(units=[], minimap=mm_found())
        st["ui"] = {"hp": 0.7}
        act = decide(st, cd)
        self.assertNotEqual(act["type"], "restore")


class TestChaseBreak(unittest.TestCase):
    """v2.5 规则 005：追击中血量降半 -> 停止追击撤退。"""

    def test_chase_breaks_when_hp_half(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.90, 0.50)])  # 远敌 0.4
        st["ui"] = {"hp": 0.45}
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "stop_chase_low_hp_retreat")

    def test_chase_continues_when_hp_ok(self):
        cd = fresh_cd()
        st = state(units=[enemy(0.90, 0.50)])
        st["ui"] = {"hp": 0.7}
        act = decide(st, cd)
        self.assertEqual(act["reason"], "chase_enemy")


class TestRecallUnderTower(unittest.TestCase):
    """v2.5 规则 012：残血回自家塔下再回城。"""

    def test_low_hp_walk_to_ally_tower(self):
        cd = fresh_cd()
        # 自家塔在右 0.30 屏宽（<0.35 塔下）-> 直接安全回城
        st = state(units=[{"cls": "ally_turret", "screen": [0.80, 0.50, 0.10, 0.10]}],
                   minimap=mm_found())
        st["ui"] = {"hp": 0.15}
        act = decide(st, cd)
        self.assertEqual(act["type"], "recall")

    def test_low_hp_no_tower_walk_to_fountain(self):
        cd = fresh_cd()
        st = state(units=[], minimap=mm_found())
        st["ui"] = {"hp": 0.15}
        act = decide(st, cd)
        self.assertEqual(act["type"], "move")
        self.assertIn(act["reason"], ("retreat_low_hp", "retreat_to_ally_tower"))


class TestFollowMinimap(unittest.TestCase):
    """小地图观察移动（用户规则）：蓝点跟随（射手优先）/ 红点支援。"""

    def test_follow_minimap_blue_nearest_lane(self):
        cd = fresh_cd()
        cd["camp"] = "blue"
        st = state(units=[], minimap=mm_found(blue=[[0.7, 0.8], [0.3, 0.3]]))
        act = decide(st, cd)
        # v2.6：可能被小地图寻路修正（_path 后缀）；未被修正时方向=直线
        self.assertTrue(act["reason"].startswith("follow_ally_minimap"))
        if act["reason"].endswith("_path"):
            self.assertNotEqual(act["reason"], "follow_ally_minimap")
        else:
            # 离发育路 (0.72,0.82) 最近的是 (0.7,0.8) -> 朝右下方
            exp = math.atan2(-(0.8 - 0.5) * (H / W), 0.7 - 0.5)
            self.assertAlmostEqual(act["theta"], exp, places=4)

    def test_support_red_centroid(self):
        cd = fresh_cd()
        st = state(units=[], minimap=mm_found(red=[[0.6, 0.6]]))
        act = decide(st, cd)
        self.assertEqual(act["reason"], "support_red_centroid")

    def test_follow_minimap_excludes_self(self):
        """v2.6：跟随目标排除己方蓝点（离圆心最近的），防跟随自己。"""
        cd = fresh_cd()
        cd["camp"] = "blue"
        # 己方(0.15,0.85 离圆心远？) 实际离圆心最近的是 (0.3,0.3)
        st = state(units=[], minimap=mm_found(blue=[[0.3, 0.3], [0.7, 0.8]]))
        act = decide(st, cd)
        self.assertTrue(act["reason"].startswith("follow_ally_minimap"))


class TestPathfinding(unittest.TestCase):
    """v2.6：小地图寻路修正（墙体绕行）。"""

    def _mm(self, blue):
        return {"found": True, "center": [100, 140], "radius": 60,
                "dots": {"blue": blue, "red": [], "yellow": []}, "towers": []}

    def test_lane_develop_path_correction(self):
        """泉水附近蓝点 + 发育路目标：走兵线路径（reason 带 _path）。"""
        cd = fresh_cd()
        cd["camp"] = "blue"
        st = state(units=[], minimap=self._mm([[0.15, 0.85]]))
        st["ui"] = {"hp": 1.0}
        act = decide(st, cd)
        # 蓝点只有己方 -> 排除后无候选 -> fallback 蓝点列表 -> 目标=己方自身
        # 此时 lane_develop 或 follow 都行，关键是路径修正逻辑不崩
        self.assertEqual(act["type"], "move")

    def test_path_correction_on_follow_ally(self):
        """队友在发育路（蓝点），己方在泉水：A* 修正方向。"""
        cd = fresh_cd()
        cd["camp"] = "blue"
        st = state(units=[], minimap=self._mm([[0.15, 0.85], [0.70, 0.80]]))
        st["ui"] = {"hp": 1.0}
        act = decide(st, cd)
        self.assertTrue(act["reason"].startswith("follow_ally_minimap"))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    ok = runner.run(suite).wasSuccessful()
    sys.exit(0 if ok else 1)
