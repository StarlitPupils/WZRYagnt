# -*- coding: utf-8 -*-
"""自我进化学习器 (v11.0): 从采集的 (状态->操作->得分) 学习改进决策。

学习目标: 在每种"状态特征"下, 哪个操作 reason 平均得分最高 -> 生成策略表:
  状态(离散桶) -> [reason -> avg_delta]  (得分变化 = 操作后的收益)
产出 configs/evolved_policy.json:
  {
    "buckets": { "state_key": {"reason": avg_delta, ...} },
    "best_reason_by_state": { "state_key": "best_reason" },
  }
optimizer.evaluate 读取: 若当前状态落在桶内 -> 该 reason 的标定分加权
自我进化: 每次学习后策略更新 -> 下次决策偏向高分操作。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> E:\WZRYagent
OUT = ROOT / "configs" / "evolved_policy.json"

# 状态关键特征桶化 (离散化): hp_bin, n_enemy, n_red, n_blue, 近敌, dead
FEATKEYS = ["hp_bin", "n_enemy_scr", "n_red_mm", "n_blue_mm", "dead"]


def bucket(rec):
    """状态 -> 桶键 (离散化特征)。"""
    hp = rec.get("hp")
    hp_bin = "low" if hp is None else ("mid" if hp < 0.35 else ("ok" if hp < 0.6 else "high"))
    n_enemy = int(rec.get("n_enemy_scr", 0) or 0)
    n_red = int(rec.get("n_red_mm", 0) or 0)
    n_blue = int(rec.get("n_blue_mm", 0) or 0)
    dead = bool(rec.get("dead"))
    return f"hp={hp_bin}|en={min(n_enemy,4)}|red={min(n_red,5)}|blue={min(n_blue,4)}|d={int(dead)}"


def learn():
    """读 selfplay 数据, 学"状态->操作->平均得分变化"策略表。"""
    d = ROOT / "data" / "selfplay"
    if not d.exists():
        print("no selfplay data")
        return
    by_state = defaultdict(lambda: defaultdict(list))   # state -> reason -> [delta]
    n = 0
    for f in sorted(d.glob("*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            reason = rec.get("reason") or rec.get("action_type") or "?"
            delta = float(rec.get("delta", 0.0))
            evt = rec.get("event")
            # 得分归属: 若有 event(本步得分) 则 delta=event.delta, 更真实
            if evt and evt.get("delta"):
                delta = float(evt["delta"])
            state_key = bucket(rec)
            by_state[state_key][reason].append(delta)
            n += 1
    if n < 10:
        print(f"数据不足({n}), 需累积更多局")
        return
    # 建策略表: 每状态选平均得分最高的 reason
    policy = {"version": "v11", "n": n, "buckets": {}, "best_reason_by_state": {}}
    best_cnt = 0
    for sk, reasons in by_state.items():
        agg = {}
        for r, ds in reasons.items():
            if len(ds) >= 2:   # 至少2样本
                agg[r] = round(sum(ds) / len(ds), 3)
        if agg:
            best = max(agg, key=agg.get)
            policy["buckets"][sk] = agg
            policy["best_reason_by_state"][sk] = best
            best_cnt += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(policy, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"自我进化: {n} 条记录, {best_cnt} 个状态桶, 策略表 -> configs/evolved_policy.json")
    # 展示前几个桶
    for sk, best in list(policy["best_reason_by_state"].items())[:8]:
        print(f"  {sk} -> best: {best}")


if __name__ == "__main__":
    learn()
