# -*- coding: utf-8 -*-
"""v3.0 行为标签（理解层）+ 反馈层奖惩 单元测试。

运行：
    venv\\Scripts\\python.exe scripts\\behavior_test.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wzry.understanding.behavior import BehaviorTagger  # noqa: E402
from wzry.feedback.reward import RewardSystem, EVENT_SCORES  # noqa: E402

T = 1000.0


def st(hp=1.0, enemies=None, turrets=None, minions=None, mm_red=None, t=T):
    units = []
    for e in (enemies or []):
        units.append({"cls": "enemy_hero", "screen": e})
    for tu in (turrets or []):
        units.append({"cls": "enemy_turret", "screen": tu})
    for m in (minions or []):
        units.append({"cls": "enemy_minion", "screen": m})
    d = {"t": t, "screen_size": [1280, 720], "units": units,
         "minimap": {"found": True, "dots": {"blue": [], "red": mm_red or []}},
         "ui": {}}
    if hp is not None:
        d["ui"]["hp"] = hp
    return d


def cd():
    return {"recall_t": 0.0, "skill2_t": 0.0, "hook_pending": 0.0}


class TestBehaviorLabel(unittest.TestCase):
    def test_moving_label(self):
        tg = BehaviorTagger()
        r = tg.update(st(), cd(), {"type": "move", "theta": 0.0}, now=T)
        self.assertEqual(r["label"], "moving")

    def test_attacking_enemy(self):
        tg = BehaviorTagger()
        r = tg.update(st(enemies=[[0.55, 0.5]]), cd(),
                      {"type": "skill", "id": 2}, now=T)
        self.assertEqual(r["label"], "attacking_enemy")

    def test_clearing_minions(self):
        tg = BehaviorTagger()
        r = tg.update(st(minions=[[0.6, 0.5]]), cd(),
                      {"type": "skill", "id": 1}, now=T)
        self.assertEqual(r["label"], "clearing_minions")

    def test_recalling_label(self):
        tg = BehaviorTagger()
        c = cd()
        c["recall_t"] = T - 3.0
        r = tg.update(st(), c, {"type": "none"}, now=T)
        self.assertEqual(r["label"], "recalling")

    def test_being_attacked(self):
        tg = BehaviorTagger()
        tg.update(st(hp=0.9), cd(), {"type": "move"}, now=T)
        r = tg.update(st(hp=0.7), cd(), {"type": "move"}, now=T + 0.1)
        self.assertTrue(r["being_attacked"])
        self.assertEqual(r["label"], "being_attacked")

    def test_dead_after_hp_missing(self):
        tg = BehaviorTagger()
        ev = None
        for i in range(10):
            r = tg.update(st(hp=None), cd(), {"type": "move"}, now=T + i * 0.5)
            if r["event"]:
                ev = r["event"]
        self.assertEqual(r["label"], "dead")
        self.assertEqual(ev, "died")

    def test_dead_only_once(self):
        tg = BehaviorTagger()
        events = []
        for i in range(30):
            r = tg.update(st(hp=None), cd(), {"type": "move"}, now=T + i * 0.5)
            if r["event"]:
                events.append(r["event"])
        self.assertEqual(events.count("died"), 1)

    def test_hook_event_after_window(self):
        tg = BehaviorTagger()
        # combo 动作 -> pending；3s 后无击杀 -> hook 事件
        r = tg.update(st(), cd(), {"type": "combo"}, now=T)
        self.assertIsNone(r["event"])
        r = tg.update(st(), cd(), {"type": "none"}, now=T + 3.5)
        self.assertEqual(r["event"], "hook")

    def test_hook_kill_within_window(self):
        tg = BehaviorTagger()
        tg.update(st(enemies=[[0.55, 0.5]]), cd(), {"type": "combo"}, now=T)
        # 1s 后近处敌人消失（被杀）
        r = tg.update(st(enemies=[]), cd(), {"type": "none"}, now=T + 1.0)
        self.assertEqual(r["event"], "hook_kill")

    def test_kill_without_hook(self):
        """攻击动作中近处敌人消失 -> 击杀。"""
        tg = BehaviorTagger()
        tg.update(st(enemies=[[0.55, 0.5]]), cd(), {"type": "skill", "id": 2}, now=T)
        r = tg.update(st(enemies=[]), cd(), {"type": "skill", "id": 2}, now=T + 1.0)
        self.assertEqual(r["event"], "kill")

    # ---------- v3.2 补充事件 ----------

    def test_assist_when_no_attack(self):
        """敌人消失但自己无攻击动作 -> 助攻。"""
        tg = BehaviorTagger()
        tg.update(st(enemies=[[0.45, 0.5]]), cd(), {"type": "move"}, now=T)
        r = tg.update(st(enemies=[]), cd(), {"type": "move"}, now=T + 1.0)
        self.assertEqual(r["event"], "assist")

    def test_be_attacked_event(self):
        """血量下降 -> 被攻击事件。"""
        tg = BehaviorTagger()
        tg.update(st(hp=0.9), cd(), {"type": "move"}, now=T)
        r = tg.update(st(hp=0.7), cd(), {"type": "move"}, now=T + 0.1)
        self.assertEqual(r["event"], "be_attacked")

    def test_be_tower_attacked_event(self):
        """塔可见 + 血量下降 -> 被防御塔攻击事件。"""
        tg = BehaviorTagger()
        tg.update(st(hp=0.9, turrets=[[0.6, 0.5]]), cd(), {"type": "move"}, now=T)
        r = tg.update(st(hp=0.6, turrets=[[0.6, 0.5]]), cd(), {"type": "move"}, now=T + 0.1)
        self.assertEqual(r["event"], "be_tower_attacked")

    def test_minion_clear_event(self):
        """近处敌兵消失 -> 清兵事件（带数量）。"""
        tg = BehaviorTagger()
        tg.update(st(minions=[[0.55, 0.5], [0.6, 0.5], [0.58, 0.52]]), cd(),
                  {"type": "skill", "id": 1}, now=T)
        r = tg.update(st(minions=[[0.6, 0.5]]), cd(), {"type": "skill", "id": 1}, now=T + 0.5)
        self.assertEqual(r["event"], "minion_clear")
        self.assertEqual(r["meta"].get("count"), 2)

    def test_recall_interrupted_event(self):
        """回城读条中被攻击 -> 回城被打断事件。"""
        tg = BehaviorTagger()
        c = cd()
        c["recall_t"] = T - 2.0
        tg.update(st(hp=0.9), c, {"type": "none"}, now=T)
        r = tg.update(st(hp=0.5), c, {"type": "none"}, now=T + 0.1)
        self.assertEqual(r["event"], "recall_interrupted")

    def test_tower_kill_event(self):
        """近处敌方塔消失 -> 推塔事件。"""
        tg = BehaviorTagger()
        tg.update(st(turrets=[[0.55, 0.5]]), cd(), {"type": "move"}, now=T)
        r = tg.update(st(turrets=[]), cd(), {"type": "move"}, now=T + 1.0)
        self.assertEqual(r["event"], "tower_kill")


class TestRewardSystem(unittest.TestCase):
    def test_scores(self):
        self.assertEqual(EVENT_SCORES["hook"], 10)
        self.assertEqual(EVENT_SCORES["kill"], 20)
        self.assertEqual(EVENT_SCORES["hook_kill"], 30)
        self.assertEqual(EVENT_SCORES["died"], -50)
        self.assertEqual(EVENT_SCORES["victory"], 80)
        self.assertEqual(EVENT_SCORES["defeat"], -80)
        # v3.2 补充系数
        self.assertEqual(EVENT_SCORES["be_tower_attacked"], -20)
        self.assertEqual(EVENT_SCORES["be_attacked"], -3)
        self.assertEqual(EVENT_SCORES["tower_kill"], 10)
        self.assertEqual(EVENT_SCORES["assist"], 15)
        self.assertEqual(EVENT_SCORES["recall_interrupted"], -10)
        self.assertEqual(EVENT_SCORES["minion_clear"], 2)
        self.assertEqual(EVENT_SCORES["supporting"], 0)

    def test_minion_clear_count_multiplier(self):
        """清兵 count 倍率：一次清 3 只 -> +6。"""
        rw = RewardSystem()
        sc = rw.on_event("minion_clear", meta={"count": 3}, now=100.0)
        self.assertEqual(sc, 6.0)

    def test_accumulate(self):
        rw = RewardSystem()
        rw.on_event("hook", now=100.0)
        rw.on_event("kill", now=101.0)
        self.assertEqual(rw.total, 30.0)

    def test_hook_kill_sequence(self):
        rw = RewardSystem()
        rw.on_event("hook", now=100.0)
        rw.on_event("hook_kill", now=101.5)
        self.assertEqual(rw.total, 40.0)

    def test_dedup_window(self):
        rw = RewardSystem()
        rw.on_event("hook", now=100.0)
        sc = rw.on_event("hook", now=100.5)  # 2s 窗口内重复
        self.assertEqual(sc, 0.0)
        self.assertEqual(rw.total, 10.0)
        sc = rw.on_event("hook", now=103.0)  # 窗口外
        self.assertEqual(sc, 10.0)

    def test_dead_and_victory(self):
        rw = RewardSystem()
        rw.on_event("died", now=100.0)
        rw.on_event("victory", now=600.0)
        self.assertEqual(rw.total, 30.0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    ok = runner.run(suite).wasSuccessful()
    sys.exit(0 if ok else 1)
