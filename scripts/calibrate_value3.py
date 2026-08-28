# -*- coding: utf-8 -*-
"""收益标定 v3 (查表式): reason -> 事后平均收益。最直接"哪步收益多大"。
决策时: 候选动作 reason 查表得收益 -> 分数 = 标定收益。语义直白, 不受线性混杂。"""
import json
import bisect
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "action_value.json"


def load_match(mdir):
    states, actions = [], []
    try:
        for ln in (mdir / "states.jsonl").read_text(encoding="utf-8").splitlines():
            states.append(json.loads(ln))
        for ln in (mdir / "actions.jsonl").read_text(encoding="utf-8").splitlines():
            actions.append(json.loads(ln))
    except Exception:
        pass
    return states, actions


def calibrate():
    byreason = defaultdict(list)
    total = 0
    for name in ["20260819_141917_889704", "20260818_034349_377946",
                 "20260819_173809_079263"]:
        mdir = ROOT / "data" / "matches" / name
        if not mdir.exists():
            continue
        states, actions = load_match(mdir)
        if not states:
            continue
        st_by_t = sorted(states, key=lambda s: s.get("t", 0))
        act_by_t = sorted(actions, key=lambda a: a.get("t", 0))
        times = [s["t"] for s in st_by_t]
        hpS = [float((s.get("ui") or {}).get("hp") or 1.0) for s in st_by_t]
        enemyC = [sum(1 for u in (s.get("units") or []) if u.get("cls") == "enemy_hero")
                  for s in st_by_t]
        for act in act_by_t:
            t = act.get("t", 0)
            i = bisect.bisect_left(times, t)
            if i >= len(st_by_t):
                continue
            j = min(bisect.bisect_right(times, t + 6.0), len(st_by_t) - 1)
            y = 0.0
            hp0, hp1 = hpS[i], hpS[j]
            if hp1 - hp0 < -0.2:
                y -= 2.0
            if hp1 - hp0 > 0.15:
                y += 0.8
            e0, e1 = enemyC[i], enemyC[j]
            if e0 > e1:
                y += 1.5
            if hp1 < 0.05 and hp0 > 0.2:
                y -= 4.0
            byreason[act.get("reason", "?")].append(y)
            total += 1
    # 查表: reason -> avg (n>=5), 否则回退 0 (中性)
    table = {}
    for k, v in byreason.items():
        if len(v) >= 5:
            table[k] = round(sum(v) / len(v), 3)
    # 默认/未知 reason 收益 0
    print(f"标定 {total} 步, {len(table)} 个 reason 有 n>=5 收益表")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} {v:+.3f} n={len(byreason[k])}")
    out = {"version": "v3-table", "reason_value": table, "default": 0.0}
    CONFIG.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已保存 configs/action_value.json")


if __name__ == "__main__":
    calibrate()
