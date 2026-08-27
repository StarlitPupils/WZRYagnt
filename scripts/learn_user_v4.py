# -*- coding: utf-8 -*-
"""learn_user_v4: 节奏/态势/习惯 学习器（两局手打）。
  1) 时间节奏: 开局0-2min / 前期2-6min / 中后期6min+ 的行为分布/位移率/接战率
  2) 态势响应: (红多=劣 / 蓝多=优) x HP档 -> 行为强度(位移率/接战率/蹲点率)
  3) 习惯: 死亡后行为 / 蹲点段时长分布 / 出门轨迹 / 转线段时长
"""
import json
import math
import collections
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    rows = []
    for p in sorted(glob.glob(str(ROOT / "data" / "matches" / "20260826_0*" / "states.jsonl"))):
        for l in Path(p).read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            if d.get("phase") != "in_match":
                continue
            mm = d.get("minimap") or {}
            if not mm.get("found"):
                continue
            dots = mm.get("dots") or {}
            greens = dots.get("green") or []
            if not greens:
                continue
            rows.append({"t": d["t"], "g": greens[0],
                         "blue": dots.get("blue") or [], "red": dots.get("red") or [],
                         "beh": (d.get("behavior") or {}).get("label", ""),
                         "dead": (d.get("behavior") or {}).get("dead", False),
                         "event": (d.get("behavior") or {}).get("event")})
    return rows


def seg_of(rows, W=3.0):
    segs = []
    i = 0
    while i < len(rows):
        t0 = rows[i]["t"]
        j = i
        while j < len(rows) and rows[j]["t"] - t0 < W:
            j += 1
        pts = rows[i:j]
        i = j
        if len(pts) < 4:
            continue
        g0, g1 = pts[0]["g"], pts[-1]["g"]
        disp = math.hypot(g1[0] - g0[0], g1[1] - g0[1])
        dead = any(p["dead"] for p in pts)
        r0 = pts[0]["red"] or []
        b0 = pts[0]["blue"] or []
        segs.append({"t": t0, "disp": disp, "dead": dead,
                     "n_red": len(r0), "n_blue": len(b0),
                     "d_red": (min([math.hypot(p[0]-g0[0], p[1]-g0[1]) for p in r0], default=9.9)
                               if r0 else 9.9),
                     "beh": collections.Counter(p["beh"] for p in pts).most_common(1)[0][0]})
    return segs


def main():
    rows = load()
    t0 = rows[0]["t"]
    for r in rows:
        r["rel"] = r["t"] - t0
    segs = seg_of(rows)
    t0s = rows[0]["t"]
    for s in segs:
        s["rel"] = s["t"] - t0s
    tot = len(segs)

    # ---- 1) 时间节奏 ----
    print("=== 1) 时间节奏 (分布%) ===")
    phases = {"0-2min(开局)": (0, 120), "2-6min(前期)": (120, 360), "6min+(中后期)": (360, 1e9)}
    for name, (a, b) in phases.items():
        ss = [s for s in segs if a <= s["rel"] < b]
        if not ss:
            continue
        mv = sum(1 for s in ss if s["disp"] > 0.02) / len(ss)
        st = sum(1 for s in ss if s["disp"] < 0.015) / len(ss)
        eng = sum(1 for s in ss if s["d_red"] < 0.35) / len(ss)
        print(f"  {name}: 段{len(ss)} 移动率{mv*100:.0f}% 蹲点率{st*100:.0f}% 接战率(红近){eng*100:.0f}%")

    # ---- 2) 态势响应 ----
    print("\n=== 2) 态势响应 (红多=劣/蓝多=优) ===")
    for st_name, cond in (("我方占优(蓝>红)", lambda s: s["n_blue"] > s["n_red"]),
                          ("均势", lambda s: s["n_blue"] == s["n_red"]),
                          ("我方劣势(红>蓝)", lambda s: s["n_blue"] < s["n_red"])):
        ss = [s for s in segs if cond(s)]
        if not ss:
            continue
        mv = sum(1 for s in ss if s["disp"] > 0.02) / len(ss)
        st = sum(1 for s in ss if s["disp"] < 0.015) / len(ss)
        eng = sum(1 for s in ss if s["d_red"] < 0.35) / len(ss)
        print(f"  {st_name}: 段{len(ss)} 移动率{mv*100:.0f}% 蹲点率{st*100:.0f}% 接战率{eng*100:.0f}%")

    # ---- 3) 习惯 ----
    print("\n=== 3) 习惯 ===")
    # 蹲点段时长分布
    hold_runs = []
    cur = 0
    prev_t = None
    for s in segs:
        if s["disp"] < 0.015 and not s["dead"]:
            if prev_t is None or s["t"] - prev_t > 5.0:
                cur = 0
            cur += 1
            prev_t = s["t"]
            hold_runs.append(cur)
        else:
            prev_t = None
    if hold_runs:
        print(f"  连续蹲点: 最长{max(hold_runs)}段(≈{max(hold_runs)*3}s), 平均{sum(hold_runs)/len(hold_runs):.1f}段")
    # 死亡后 10s 行为
    dies = [i for i, s in enumerate(segs) if s["dead"]]
    post = [s for i, s in enumerate(segs) if i > 0 and segs[i-1]["dead"] and not s["dead"]]
    print(f"  死亡段 {len(dies)}; 复活后首段位移: " +
          ", ".join(f"{s['disp']:.2f}" for s in post[:8]))
    # 出门轨迹(开局前30s)
    ops = [s for s in segs if s["rel"] < 30]
    print(f"  开局30s: 段{len(ops)} 平均位移 {sum(s['disp'] for s in ops)/max(1,len(ops)):.2f}")


if __name__ == "__main__":
    main()
