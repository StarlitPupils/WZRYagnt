# -*- coding: utf-8 -*-
"""收益标定v2: 修正死亡惩罚(击杀+2但死亡-4更重) + 字符约束特征方向。"""
import json
import bisect
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "action_value.json"
FEATS = ["hp", "n_enemy", "n_red", "n_blue", "n_turret", "d_turret", "in_turret_zone"]


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


def feat_vec(st):
    ui = st.get("ui") or {}
    hp = float(ui.get("hp") or 1.0)
    dots = (st.get("minimap") or {}).get("dots") or {}
    n_red = len(dots.get("red") or [])
    n_blue = len(dots.get("blue") or [])
    units = st.get("units") or []
    n_enemy = sum(1 for u in units if u.get("cls") == "enemy_hero")
    turrets = [u for u in units if u.get("cls") == "enemy_turret"]
    n_turret = len(turrets)
    d_turret = 1.0
    if turrets:
        s = turrets[0].get("screen") or [0.5, 0.5]
        d_turret = math.hypot(s[0] - 0.5, s[1] - 0.5)
    in_zone = 1.0 if (n_turret and d_turret < 0.45) else 0.0
    return [hp, n_enemy, n_red, n_blue, n_turret, d_turret, in_zone]


def calibrate():
    rows = []
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
            x = feat_vec(st_by_t[i])
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
            # v2 死亡重罚: hp 降至 0 (死亡) = -4
            if hp1 < 0.05 and hp0 > 0.2:
                y -= 4.0
            rows.append((x, y, act.get("reason", "")))
    if not rows:
        print("无数据")
        return
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    lam = 1e-2
    coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    intercept = float(y.mean() - X.mean(axis=0) @ coef)
    print(f"标定样本 {len(rows)}")
    print("特征权重:", {f: round(float(c), 3) for f, c in zip(FEATS, coef)})
    byreason = defaultdict(list)
    for _, yv, reason in rows:
        byreason[reason].append(yv)
    print("\n--- reason 事后收益 ---")
    for k, v in sorted(byreason.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        if len(v) >= 5:
            print(f"  {k:40s} avg={sum(v)/len(v):+.2f} n={len(v)}")
    out = {"feats": FEATS, "coef": [round(float(c), 4) for c in coef],
           "intercept": round(float(intercept), 4)}
    CONFIG.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已保存 configs/action_value.json")


if __name__ == "__main__":
    calibrate()
