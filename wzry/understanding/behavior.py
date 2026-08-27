# -*- coding: utf-8 -*-
"""理解层：行为标签（用户规则 v3.0）。

理解层在感知结果之上输出"当前行为标签"，让 agent 知道自己在做什么，
并驱动反馈层奖惩（wzry.feedback.reward）与决策优化。

行为标签（状态类，每帧输出一个主标签 label）：
  - dead                我已阵亡（等待泉水复活）
  - recalling           我正在回城
  - being_tower_attacked 我被防御塔攻击了
  - being_attacked      我被攻击了
  - attacking_enemy     我在攻击敌方英雄
  - clearing_minions    我在清理兵线
  - pushing_tower       我在推塔
  - supporting          我在支援队友
  - moving              我正在移动
  - idle                待机/无动作

事件（event，触发反馈层计分）：
  - hook      勾到人（decide 判定 combo = 二技能勾中拉近）
  - died      阵亡（自己血条持续缺失）
  - kill/hook_kill/victory/defeat 等由外部信号触发（后续接入）

输入：state_dict（感知聚合结果）+ cooldowns（决策记忆）+ action（本帧决策）
"""
import time

DEAD_MISS_FRAMES = 6     # 自己血条连续缺失帧数 -> 疑似阵亡
DEAD_CONFIRM_S = 3.0     # 缺失持续秒数 -> 确认阵亡（防复活/画面抖动误判）
DEAD_BANNER_MIN = 2500   # v2.87 死亡横幅暗红px阈值(真死>=2500, 击杀播报<=1800)
DEAD_BANNER_STREAK = 2   # 横幅连续帧数 -> 铁证死亡(死亡时屏幕有'查看死亡回放'提示)
KILL_BANNER_MIN = 900    # v2.96 击杀播报金条px阈值(金色'你击败了'横幅)
KILL_WINDOW_S = 8.0      # 击杀播报窗口: 窗口内敌人消失才算击杀/助攻
RECALL_ACTIVE_S = 8.0    # 回城读条持续秒数（与决策层一致）
HOOK_KILL_WINDOW_S = 3.0  # 勾到人后击杀确认窗口（勾杀组合 +30）
KILL_NEAR_FRAC = 0.35     # 击杀判定：近处敌人（<0.35 屏宽）从视野消失
ASSIST_NEAR_FRAC = 0.60   # 助攻判定：0.35-0.6 屏宽敌人消失（队友击杀）
MINION_NEAR_FRAC = 0.50   # 清兵判定：近处敌兵消失
TOWER_NEAR_FRAC = 0.60    # 推塔判定：近处敌方塔消失
HP_DROP_FRAC = 0.06       # 被攻击判定：血量下降阈值（v3.2 实测 0.03 噪声误报，提至 0.06）


class BehaviorTagger:
    """从感知/决策状态派生行为标签与事件。"""

    def __init__(self):
        self._hp_miss_streak = 0
        self._hp_miss_since = 0.0
        self._last_hp = None
        self._dead_reported = False
        self._dead_since = None      # v2.87 死亡(横幅铁证)起始时间
        self._banner_streak = 0      # v2.87 死亡横幅连续帧数
        self._kill_banner_t = None   # v2.96 击杀播报金条时间(窗口内击杀才成立)
        self._kill_banner_seen = False
        self._last_label = None
        self._last_near_enemies = 0
        self._last_near_minions = 0
        self._last_near_turrets = 0
        self._pending_hook = 0.0   # 勾到人时间（延迟确认：3s 内击杀则计 hook_kill）
        self._recall_interrupted_reported = False

    def reset_match(self):
        """v2.95 新对局开始: 清零死亡/横幅残留(上局死亡 _dead_since 40s内会污染新局 ->
        开局 hp=None + 残留 -> 假 died -50)"""
        self._hp_miss_streak = 0
        self._hp_miss_since = 0.0
        self._last_hp = None
        self._dead_reported = False
        self._dead_since = None
        self._banner_streak = 0
        self._kill_banner_t = None
        self._kill_banner_seen = False
        self._last_near_enemies = 0
        self._last_near_minions = 0
        self._last_near_turrets = 0
        self._pending_hook = 0.0
        self._recall_interrupted_reported = False

    def update(self, state_dict, cooldowns, action, now=None) -> dict:
        """每帧调用。返回 {"label", "event", "dead", ...}。"""
        now = now if now is not None else float(state_dict.get("t") or time.time())
        ui = state_dict.get("ui") or {}
        hp = ui.get("hp")
        units = state_dict.get("units") or []
        enemies = [u for u in units if str(u.get("cls", "")) == "enemy_hero"]
        turrets = [u for u in units if str(u.get("cls", "")) == "enemy_turret"]
        enemy_minions = [u for u in units if str(u.get("cls", "")) == "enemy_minion"]
        minimap = state_dict.get("minimap") or {}
        mm_red = (minimap.get("dots") or {}).get("red") or [] if minimap.get("found") else []

        def dist0(u):
            s = u.get("screen") or [0.5, 0.5, 0, 0]
            return ((s[0] - 0.5) ** 2 + (s[1] - 0.5) ** 2) ** 0.5

        # ---- 阵亡检测 v2.87: 死亡横幅铁证优先(用户: 死亡时屏幕有提示) ----
        # 屏幕死亡横幅("查看死亡回放"暗红)连续2帧 -> 铁证死亡;
        # 血条缺失仅作辅助(血条检测可能丢失, 横幅缺失时不判死)
        banner_px = int(ui.get("death_banner") or 0)
        self._banner_streak = (self._banner_streak + 1
                               if banner_px >= DEAD_BANNER_MIN else 0)
        banner_dead = self._banner_streak >= DEAD_BANNER_STREAK
        if banner_dead:
            self._dead_since = now
        dead = banner_dead
        if hp is None:
            if self._hp_miss_streak == 0:
                self._hp_miss_since = now
            self._hp_miss_streak += 1
        else:
            self._hp_miss_streak = 0
            self._hp_miss_since = 0.0
        # 血条缺失辅助: 仅当横幅曾出现过(近期死亡回忆)才补充确认
        miss_dead = (self._hp_miss_streak >= DEAD_MISS_FRAMES
                     and now - self._hp_miss_since >= DEAD_CONFIRM_S
                     and self._dead_since is not None
                     and now - self._dead_since < 40.0)
        dead = dead or miss_dead
        # 复活: 血条恢复 + 横幅消失 -> 解除
        if (self._dead_since is not None and hp is not None
                and banner_px < DEAD_BANNER_MIN
                and now - self._dead_since > 3.0):
            self._dead_since = None
            self._banner_streak = 0
        if not dead:
            self._dead_reported = False

        # ---- 勾到人（延迟确认）：本帧 combo 决策 = 勾中拉近 ----
        event = None
        meta = {}
        if action.get("type") == "combo":
            self._pending_hook = now if not self._pending_hook else self._pending_hook

        # ---- 敌人消失：击杀/助攻/勾杀 (v2.96 需击杀播报铁证, 旧"消失=击杀"是假事件) ----
        # 击杀播报金条 >=KILL_BANNER_MIN 记入 _kill_banner_t 窗口 8s; 窗口内敌人消失才算击杀类
        _kb = int(ui.get("kill_banner") or 0)
        if _kb >= KILL_BANNER_MIN:
            self._kill_banner_t = now
            self._kill_banner_seen = True
        kill_confirm = (self._kill_banner_t is not None
                        and now - self._kill_banner_t <= KILL_WINDOW_S)
        near_now = sum(1 for u in enemies if dist0(u) < KILL_NEAR_FRAC)
        assist_now = sum(1 for u in enemies if KILL_NEAR_FRAC <= dist0(u) < ASSIST_NEAR_FRAC)
        if (self._last_near_enemies > near_now and not dead):
            if kill_confirm:
                if self._pending_hook:
                    event = "hook_kill"     # 勾到人并击杀 +30
                    self._pending_hook = 0.0
                elif action.get("type") in ("skill", "attack", "combo"):
                    event = "kill"          # 自己有攻击动作 -> 击杀 +20
                else:
                    event = "assist"        # 附近敌人消失且自己未输出 -> 助攻 +15
                self._kill_banner_t = None   # 只消费一次
            else:
                event = None                # 无敌方击杀播报=敌人走掉/进草, 非击杀
        elif self._pending_hook and now - self._pending_hook > HOOK_KILL_WINDOW_S:
            event = "hook"              # 勾到人但未造成击杀
            self._pending_hook = 0.0
        self._last_near_enemies = near_now

        # ---- 清兵：近处敌方小兵消失（每个 +2）----
        minions_now = sum(1 for u in enemy_minions if dist0(u) < MINION_NEAR_FRAC)
        cleared = max(0, self._last_near_minions - minions_now)
        if cleared > 0 and event is None and not dead:
            event = "minion_clear"
            meta["count"] = cleared
        self._last_near_minions = minions_now

        # ---- 推塔：近处敌方塔消失 ----
        turrets_now = sum(1 for u in turrets if dist0(u) < TOWER_NEAR_FRAC)
        if self._last_near_turrets > turrets_now and event is None and not dead:
            event = "tower_kill"
        self._last_near_turrets = turrets_now

        # ---- 被攻击：血量下降 ----
        being_attacked = False
        if hp is not None and self._last_hp is not None:
            if self._last_hp - hp > HP_DROP_FRAC:
                being_attacked = True
        self._last_hp = hp

        # ---- 回城被打断：回城读条中（recalling）被攻击 ----
        in_recall = now - float(cooldowns.get("recall_t", 0.0)) < RECALL_ACTIVE_S
        if in_recall and being_attacked and event is None:
            event = "recall_interrupted"
            self._recall_interrupted_reported = True
        if not in_recall:
            self._recall_interrupted_reported = False

        # ---- 被防御塔攻击：塔可见 + 血量下降 ----
        being_tower_attacked = bool(turrets) and being_attacked
        if being_tower_attacked and event is None and not dead:
            event = "be_tower_attacked"
        elif being_attacked and event is None and not dead:
            event = "be_attacked"

        # ---- 阵亡事件（最高优先级，首次确认时触发一次）----
        if dead and not self._dead_reported:
            event = "died"
            self._dead_reported = True
            self._pending_hook = 0.0

        # ---- 行为标签（优先级从高到低）----
        action_type = action.get("type", "none")
        if dead:
            label = "dead"
        elif action_type == "recall" or now - float(cooldowns.get("recall_t", 0.0)) < RECALL_ACTIVE_S:
            label = "recalling"
        elif being_tower_attacked:
            label = "being_tower_attacked"
        elif being_attacked:
            label = "being_attacked"
        elif action_type in ("skill", "attack") and enemies:
            label = "attacking_enemy"
        elif action_type in ("skill", "attack") and enemy_minions:
            label = "clearing_minions"
        elif action_type == "move" and enemies:
            label = "attacking_enemy"      # 追击敌人
        elif turrets and action_type == "move":
            label = "pushing_tower"
        elif mm_red and action_type == "move":
            label = "supporting"
        elif action_type == "move":
            label = "moving"
        elif enemy_minions:
            label = "clearing_minions"
        else:
            label = "idle"
        self._last_label = label

        return {"label": label, "event": event, "meta": meta, "dead": dead,
                "being_attacked": being_attacked,
                "being_tower_attacked": being_tower_attacked}
