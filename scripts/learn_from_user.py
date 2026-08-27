# -*- coding: utf-8 -*-
"""从用户手打对局学习：states(感知) × actions(摇杆) → 反推行为标签 → 规则表。

数据: data/demos/20260818_125747/{states.jsonl, actions.json}
方法:
  1. 对齐 state tick 与用户摇杆方向(20Hz, ±0.15s)
  2. 特征: hp / 屏幕内敌人最近距离 / 小地图敌点最近距离 / 队友距离 / 我方半场 / 兵线
  3. 意图标签(未来3s 小地图净位移 + 相对位置): 进攻/跟队/撤退/发育/待机
  4. 聚集: 特征桶 -> 主导意图 -> 学习规则表
"""
import json
import math
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demos" / "20260818_125747"
MM_SIZE = 232.0

def load_states():
    out = []
    for ln in (DEMO / "states.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("phase") != "in_match":
            continue
        out.append(d)
    return out

def load_actions():
    d = json.loads((DEMO / "actions.json").read_text(encoding="utf-8"))
    # actions t 是相对视频的秒 -> 与 states t 的对应: states 首帧 t0=1787029107.246; video t0?
    # actions 里 t 从 0.45 开始; 取 actions[0] frame=27 => video 首帧 t=1787029083.93(触摸流首个)
    # 校准: 用 frame 与 fps 推秒, 再对齐: video_t0 = states[0].t - actions[0].t
    acts = d["actions"]
    return acts

def main():
    states = load_states()
    acts = load_actions()
    # 时间对齐: 假设 states 与 actions 同源同起点(=首帧)
    t0 = states[0]["t"]
    acts_sorted = sorted(acts, key=lambda a: a["t"])
    # 转换: actions t(视频相对秒) -> 绝对: 用首帧 frame 关系: t_abs = t0 + (a["t"] - acts_sorted[0]["t"])
    a0t = acts_sorted[0]["t"]
    def act_at(t):
        best = None
        for a in acts_sorted:
            if best is None or abs(a["t"] - (t - t0 + a0t)) < abs(best["t"] - (t - t0 + a0t)):
                best = a
        return best

    DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
            "left_up": (-0.7, -0.7), "right_up": (0.7, -0.7),
            "left_down": (-0.7, 0.7), "right_down": (0.7, 0.7)}

    rows = []
    for i, s in enumerate(states):
        t = s["t"]
        ui = s.get("ui") or {}
        hp = float(ui.get("hp") or 1.0)
        mm = s.get("minimap") or {}
        dots = (mm.get("dots") or {}) if mm.get("found") else {}
        blues = dots.get("blue") or []
        reds = dots.get("red") or []
        greens = dots.get("green") or []
        # 自己位置: green 优先(旧模块探测), 否则蓝点离 (0.5,0.5) 最近? 用第一个蓝
        selfp = greens[0] if greens else (blues[0] if blues else None)
        units = s.get("units") or []
        ne_dist = 9.9
        for u in units:
            if u.get("cls") == "enemy_hero" or u.get("cls") == "enemy_minion":
                cx = u.get("screen", [0.5, 0.5])[0]
                cy = u.get("screen", [0.5, 0.5])[1]
                d = math.hypot(cx - 0.5, cy - 0.5)
                ne_dist = min(ne_dist, d)
        # 用户意图: 摇杆方向 + 未来3s蓝群位移的方向
        a = act_at(t)
        if a is None or a.get("type") != "move":
            continue
        ddir = DIRS.get(a.get("direction"))
        if ddir is None:
            continue
        # 未来 3s: 队友蓝点质心位移
        j = i
        while j < len(states) and states[j]["t"] - t < 3.0:
            j += 1
        if j >= len(states):
            j = len(states) - 1
        d2 = (states[j].get("minimap") or {}).get("dots") or {}
        blues2 = d2.get("blue") or []
        if not blues or not blues2 or not selfp:
            continue
        c1 = [sum(p[0] for p in blues) / len(blues), sum(p[1] for p in blues) / len(blues)]
        c2 = [sum(p[0] for p in blues2) / len(blues2), sum(p[1] for p in blues2) / len(blues2)]
        disp = (c2[0] - c1[0], c2[1] - c1[1])
        dispn = math.hypot(*disp) or 1e-6
        # 相对位置: 红点最近距离, 蓝点最近距离
        red_d = min([math.hypot(p[0] - selfp[0], p[1] - selfp[1]) for p in reds], default=9.9)
        blue_d = min([math.hypot(p[0] - selfp[0], p[1] - selfp[1]) for p in blues], default=9.9)
        # 意图标签
        toward_red = (disp[0] * (-reds[0][0] + selfp[0]) + disp[1] * (-reds[0][1] + selfp[1])) > 0 if reds else False
        if hp < 0.30 and red_d < 0.5:
            label = "撤退"
        elif reds and red_d < 0.35 and dispn > 0.02 and toward_red:
            label = "进攻"
        elif reds and red_d < 0.5 and dispn > 0.02:
            label = "支援"
        elif blue_d < 0.4 and dispn > 0.015:
            label = "跟队"
        elif ne_dist < 0.5:
            label = "战斗"
        elif dispn <= 0.012:
            label = "待机"
        else:
            label = "发育"
        rows.append({"t": t, "hp": round(hp, 2), "ne_dist": round(ne_dist, 2),
                     "red_d": round(red_d, 2), "blue_d": round(blue_d, 2),
                     "disp": round(dispn, 3), "dir": a.get("direction"), "label": label})

    print("有效样本:", len(rows))
    cnt = collections.Counter(r["label"] for r in rows)
    print("\n=== 行为标签分布(用户) ===")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} ({v/len(rows)*100:.0f}%)")

    # 特征桶 -> 主导意图
    print("\n=== 学习规则表 (特征桶 -> 主导意图) ===")
    buckets = collections.defaultdict(collections.Counter)
    for r in rows:
        hp_b = "HP<40%" if r["hp"] < 0.4 else ("HP40-70%" if r["hp"] < 0.7 else "HP>70%")
        ne_b = "敌近(<0.3)" if r["ne_dist"] < 0.3 else ("敌中(<0.6)" if r["ne_dist"] < 0.6 else "敌远")
        red_b = "红近(<0.3)" if r["red_d"] < 0.3 else ("红中(<0.5)" if r["red_d"] < 0.5 else "红远")
        buckets[(hp_b, ne_b, red_b)][r["label"]] += 1
    for k, v in sorted(buckets.items()):
        tot = sum(v.values())
        if tot >= 5:
            top = v.most_common(3)
            print(f"  {k}: " + ", ".join(f"{a}({b/tot*100:.0f}%)" for a, b in top))

if __name__ == "__main__":
    main()
