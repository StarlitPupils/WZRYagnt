# -*- coding: utf-8 -*-
"""两局手打联合训练: 特征桶 -> 用户主导行为决策表 (learn_user_v3)。

输入: data/matches/20260826_0* states (in_match, 自己绿点轨迹 + 红蓝点)
输出:
  - 行为段(Counter)
  - 特征桶决策表: (红距, 蓝距, 位移) -> 主导标签
  - 蹲点/转线统计: 静止段分布 / 大位移段分布
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
                         "blue": dots.get("blue") or [], "red": dots.get("red") or []})
    return rows


def main():
    rows = load()
    print("采样:", len(rows), f"跨度 {rows[-1]['t']-rows[0]['t']:.0f}s")
    W = 3.0
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
        dx, dy = g1[0] - g0[0], g1[1] - g0[1]
        disp = math.hypot(dx, dy)
        r0 = pts[0]["red"] or []
        b0 = pts[0]["blue"] or []
        d_red0 = (min([math.hypot(p[0]-g0[0], p[1]-g0[1]) for p in r0], default=9.9)
                  if r0 else 9.9)
        to_red = False
        if r0:
            tr = min(r0, key=lambda p: (p[0]-g0[0])**2 + (p[1]-g0[1])**2)
            to_red = (dx*(tr[0]-g0[0]) + dy*(tr[1]-g0[1])) > 0
        to_blue = False
        if b0:
            bc = [sum(p[0] for p in b0)/len(b0), sum(p[1] for p in b0)/len(b0)]
            to_blue = (dx*(bc[0]-g0[0]) + dy*(bc[1]-g0[1])) > 0
        if disp < 0.015:
            label = "待机"
        elif to_red and d_red0 < 0.5:
            label = "支援进攻"
        elif to_blue:
            label = "跟队"
        elif d_red0 < 0.3:
            label = "警戒移动"
        else:
            label = "发育游走"
        segs.append({"t": t0, "disp": disp, "d_red": d_red0, "d_blue":
                     (min([math.hypot(p[0]-g0[0], p[1]-g0[1]) for p in b0], default=9.9)
                      if b0 else 9.9), "label": label})
    cnt = collections.Counter(s["label"] for s in segs)
    tot = len(segs)
    print(f"\n=== 行为段 ({tot}) ===")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} ({v/tot*100:.0f}%)")

    # 特征桶决策表
    print("\n=== 决策表 (红距/蓝距) -> 用户主导行为 ===")
    buckets = collections.defaultdict(collections.Counter)
    for s in segs:
        rb = "红近<0.3" if s["d_red"] < 0.3 else ("红中<0.6" if s["d_red"] < 0.6 else "红远")
        bb = "蓝近<0.2" if s["d_blue"] < 0.2 else ("蓝中<0.5" if s["d_blue"] < 0.5 else "蓝远")
        buckets[(rb, bb)][s["label"]] += 1
    for k, v in sorted(buckets.items()):
        t2 = sum(v.values())
        if t2 >= 6:
            top = v.most_common(2)
            print(f"  {k}: " + ", ".join(f"{a}({b/t2*100:.0f}%)" for a, b in top))

    # 蹲点/转线
    statics = [s for s in segs if s["label"] == "待机"]
    bigs = [s for s in segs if s["disp"] > 0.5]
    print(f"\n=== 蹲点(待机) {len(statics)} 段; 转线(>0.5) {len(bigs)} 段 ===")
    print("转线段距离:", [round(s['disp'], 2) for s in bigs[:12]])

    out = ROOT / "temp" / "user_policy_v2.json"
    out.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
    print("-> temp/user_policy_v2.json")


if __name__ == "__main__":
    main()
