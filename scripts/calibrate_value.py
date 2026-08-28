# -*- coding: utf-8 -*-
"""收益标定器 (用户铁律: 标定收益, 非学规则)。

原理: 对已有对局 states+actions 逐动作归因收益:
  - 事后 6s 窗口内 我 hp 涨 = 正 (保命/回血)
  - 6s 窗口内 我方击杀敌英(屏内敌英消失) = 大正
  - 6s 窗口内 我 hp 大降/死亡 = 大负
  - 6s 窗口内 敌塔出现且我近 = 负(被塔打没撤)
训练:
  X = 动作发生时的特征向量 (我方hp/mp, 屏内敌数, 红点数, 蓝点数, 敌塔数, 距塔距离)
  y = 事后收益 (线性近似: 击杀+2 助攻+1 掉血-1.5 死亡-3 被塔-0.5 回血+0.5)
  -> 用岭回归拟合 => 收益函数: 每种状态下"执行任意动作"的预期收益权重
保存 configs/action_value.json (特征权重), optimizer.evaluate 用它评分。
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "action_value.json"

# 特征索引
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


def calibrate(match_dirs):
    rows = []   # (feat_vec, y, reason)
    for name in match_dirs:
        mdir = ROOT / "data" / "matches" / name
        if not mdir.exists():
            continue
        states, actions = load_match(mdir)
        if not states:
            continue
        # 时间索引
        st_by_t = sorted(states, key=lambda s: s.get("t", 0))
        act_by_t = sorted(actions, key=lambda a: a.get("t", 0))
        # 重叠动作特征: 对每个 action, 找紧随的 state, 前瞻窗口收益
        # 简易: 按时刻对齐, 前向 6s
        import bisect
        times = [s["t"] for s in st_by_t]
        hpSeries = [float((s.get("ui") or {}).get("hp") or 1.0) for s in st_by_t]
        enemySeries = [sum(1 for u in (s.get("units") or []) if u.get("cls") == "enemy_hero")
                       for s in [st_by_t[i]]] if False else None
        enemy_cnt = []
        for s in st_by_t:
            enemy_cnt.append(sum(1 for u in (s.get("units") or []) if u.get("cls") == "enemy_hero"))
        turret_cnt = []
        for s in st_by_t:
            turret_cnt.append(sum(1 for u in (s.get("units") or []) if u.get("cls") == "enemy_turret"))
        for act in act_by_t:
            t = act.get("t", 0)
            i = bisect.bisect_left(times, t)
            if i >= len(st_by_t):
                continue
            st = st_by_t[i]
            x = feat_vec(st)
            # 前瞻 6s 收益
            j = bisect.bisect_right(times, t + 6.0)
            w = times[min(j, len(times) - 1)] - t
            if w < 0.1:
                w = 6.0
            y = 0.0
            hp0 = hpSeries[i]
            hp1 = hpSeries[min(j, len(hpSeries) - 1)]
            if hp1 - hp0 < -0.2:
                y -= 2.0                      # 掉血
            if hp1 - hp0 > 0.15:
                y += 0.8                       # 回血/撤出
            e0 = enemy_cnt[i]
            e1 = enemy_cnt[min(j, len(enemy_cnt) - 1)]
            if e0 > e1:
                y += 2.0                        # 击杀(屏内敌英消失)
            t0 = turret_cnt[i]
            t1 = turret_cnt[min(j, len(turret_cnt) - 1)]
            if t0 > 0 and x[6] > 0.5:
                y -= 0.5                        # 被塔压
            rows.append((x, y, act.get("reason", "")))
    if not rows:
        print("无数据")
        return
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    # 岭回归 (退化numpy手写: (X^T X + λI)^-1 X^T y)
    lam = 1e-2
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    coef = np.linalg.solve(XtX, X.T @ y)
    intercept = float(y.mean() - X.mean(axis=0) @ coef)
    print(f"标定样本 {len(rows)}")
    print("特征权重:", {f: round(float(c), 3) for f, c in zip(FEATS, coef)})
    print("intercept:", round(intercept, 3))
    # 按 reason 收益摘要
    byreason = defaultdict(list)
    for _, yv, reason in rows:
        byreason[reason].append(yv)
    print("\n--- reason 事后收益 (n>=5) ---")
    for k, v in sorted(byreason.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        if len(v) >= 5:
            print(f"  {k:40s} avg={sum(v)/len(v):+.2f} n={len(v)}")
    out = {"feats": FEATS, "coef": [round(float(c), 4) for c in coef],
           "intercept": round(float(intercept), 4)}
    CONFIG.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 configs/action_value.json")


if __name__ == "__main__":
    calibrate(["20260819_141917_889704", "20260818_034349_377946",
               "20260819_173809_079263"])
