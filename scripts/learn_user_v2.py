# -*- coding: utf-8 -*-
"""学习用户本轮手打: 绿点自轨迹 → 移动决策段 → 行为标签分布 + 与AI对比。

数据: data/matches/20260826_0*/*.jsonl (观察模式记录)
标签规则(位移+局面对比):
  - 朝最近红点净移: 支援/进攻
  - 朝蓝点质心: 跟队
  - 朝泉水(自己出生角/FOUNTAIN): 回城/撤退
  - 原地(位移<0.02/3s): 待机/发育
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
                         "beh": (d.get("behavior") or {}).get("label")})
    return rows

def main():
    rows = load()
    print("有效采样:", len(rows), f"跨度 {rows[-1]['t']-rows[0]['t']:.0f}s")
    if not rows:
        return
    # 3s 窗口分段
    W = 3.0
    segs = []
    i = 0
    while i < len(rows):
        t0 = rows[i]["t"]
        j = i
        while j < len(rows) and rows[j]["t"] - t0 < W:
            j += 1
        pts = rows[i:j]
        if len(pts) < 4:
            i = j
            continue
        g0, g1 = pts[0]["g"], pts[-1]["g"]
        dx, dy = g1[0] - g0[0], g1[1] - g0[1]
        disp = math.hypot(dx, dy)
        # 局面对比
        r0 = pts[0]["red"] or []
        b0 = pts[0]["blue"] or []
        d_red0 = (min([math.hypot(p[0]-g0[0], p[1]-g0[1]) for p in r0], default=9.9)
                  if r0 else 9.9)
        if r0:
            tr = min(r0, key=lambda p: (p[0]-g0[0])**2 + (p[1]-g0[1])**2)
            # 位移指向红点?
            to_red = (dx*(tr[0]-g0[0]) + dy*(tr[1]-g0[1])) > 0
        else:
            to_red = False
        if b0:
            bc = [sum(p[0] for p in b0)/len(b0), sum(p[1] for p in b0)/len(b0)]
            to_blue = (dx*(bc[0]-g0[0]) + dy*(bc[1]-g0[1])) > 0
        else:
            to_blue = False
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
        segs.append({"t": t0, "dx": dx, "dy": dy, "disp": disp, "label": label})
        i = j
    cnt = collections.Counter(s["label"] for s in segs)
    tot = len(segs)
    print(f"\n=== 用户移动段 ({tot} 段 x {W:.0f}s) ===")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} ({v/tot*100:.0f}%)")

    print("\n=== 用户轨迹快照(每20段) ===")
    for s in segs[::max(1, tot//14)]:
        print(f"  t={s['t']:.0f} {s['label']} disp={s['disp']:.3f} d=({s['dx']:+.2f},{s['dy']:+.2f})")

    # 送存
    out = ROOT / "temp" / "user_play_segs.json"
    out.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
    print("\n-> temp/user_play_segs.json")

if __name__ == "__main__":
    main()
