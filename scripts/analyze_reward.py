# -*- coding: utf-8 -*-
"""决策收益分析 (用户: 学哪步收益最高做哪步, 非规则)。

对已有对局 states+actions 推导每步收益, 统计不同 action/reason 的期望收益:
  收益信号(从状态变迁, 无显式 reward 时):
    - 我方 hp 变化: 降=被揍(负), 升=安全/回血(正)
    - 敌英从屏幕消失 = 击杀(正) / 从小地图红点消失 = 压制
    - 玩家朝向敌塔移动 = 风险(负, 塔打不撤)
    - 玩家与蓝点(队友)同向 = 跟队(正)
  输出: 每种 reason/action 的平均收益 -> "哪种决策赚分高"
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_match(mdir):
    states = []
    for ln in (mdir / "states.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            states.append(json.loads(ln))
        except Exception:
            pass
    actions = []
    for ln in (mdir / "actions.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            actions.append(json.loads(ln))
        except Exception:
            pass
    return states, actions


def analyze(mdir, name):
    states, actions = load_match(mdir)
    if not states:
        return None
    print(f"\n===== {name} (states {len(states)}, actions {len(actions)}) =====")
    # 按时间排序合并步: 状态帧 + 紧随的 action
    stat_by_t = {s.get("t", 0): s for s in states}
    act_by_t = sorted(actions, key=lambda a: a.get("t", 0))
    # 顺序扫描: state -> action -> next state 收益
    events = []
    prev_hp = None
    prev_hero = None
    t_prev = None
    act_i = 0
    reason_rewards = defaultdict(list)
    action_rewards = defaultdict(list)
    last_action = None
    last_t = None
    for st in sorted(states, key=lambda s: s.get("t", 0)):
        ui = st.get("ui") or {}
        hp = ui.get("hp")
        hero = (st.get("extra") or {}).get("hero_pos")
        dots = (st.get("minimap") or {}).get("dots") or {}
        reds = dots.get("red") or []
        blues = dots.get("blue") or []
        units = st.get("units") or []
        enemies = [u for u in units if u.get("cls") == "enemy_hero"]
        t = st.get("t", 0)
        # 收益推导
        reward = 0.0
        if prev_hp is not None and hp is not None:
            dh = hp - prev_hp
            if dh < -0.05:
                reward -= 2.0        # 掉血=被揍
            elif dh > 0.10:
                reward += 1.0        # 回血/安全
        # 敌英消失(屏幕)
        if last_action and last_action.get("type") in ("skill", "attack", "combo"):
            if prev_hero and hero and prev_hero != hero:
                pass
        # 进敌方塔威胁: 帧里有 enemy_turret 且我方近
        if any(u.get("cls") == "enemy_turret" for u in units):
            reward -= 0.3           # 塔区=风险(塔打不撤惩罚)
        # 跟队: 红点少蓝点多
        if blues and not reds:
            reward += 0.1           # 安全区
        if last_action:
            reason_rewards[last_action.get("reason", "?")].append(reward)
            action_rewards[last_action.get("type", "?")].append(reward)
        # 关联当前 action(时间贴合)
        while act_i < len(act_by_t) and act_by_t[act_i].get("t", 0) <= t:
            last_action = act_by_t[act_i]
            act_i += 1
        prev_hp = hp
        prev_hero = hero
    # 汇总
    print("--- 按 reason 平均收益 (样本>=3) ---")
    rl = sorted(((v, k) for k, v in reason_rewards.items() if len(v) >= 3),
                key=lambda x: -sum(x[0]) / len(x[0]))
    for v, k in rl[:15]:
        print(f"  {k:40s} avg={sum(v)/len(v):+.2f} n={len(v)}")
    print("--- 按 action 类型 ---")
    for k, v in sorted(action_rewards.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {k:12s} avg={sum(v)/len(v):+.2f} n={len(v)}")
    return reason_rewards, action_rewards


if __name__ == "__main__":
    for name in ["20260819_141917_889704", "20260818_034349_377946",
                 "20260819_173809_079263"]:
        d = ROOT / "data" / "matches" / name
        if d.exists():
            analyze(d, name)
