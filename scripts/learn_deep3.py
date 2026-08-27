# -*- coding: utf-8 -*-
"""深度学习 v3: skill_dark 序列 -> 技能按下时刻 x 同帧局面 -> 用户技能触发规则。"""
import sys
import json
import math
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "data" / "demos" / "20260818_125747"


def main():
    states = []
    for ln in (DEMO / "states.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("phase") != "in_match":
            continue
        states.append(d)
    print("ticks:", len(states))

    # 特征提取
    feats = []
    for s in states:
        ui = s.get("ui") or {}
        hsv = ui.get("skill_dark") or {}
        hp = float(ui.get("hp") or 1.0)
        units = s.get("units") or []
        e_d = 9.9
        for u in units:
            if u.get("cls") in ("enemy_hero", "enemy_minion"):
                scr = u.get("screen") or [0.5, 0.5]
                e_d = min(e_d, math.hypot(scr[0] - 0.5, scr[1] - 0.5))
        mm = s.get("minimap") or {}
        dots = (mm.get("dots") or {}) if mm.get("found") else {}
        reds = dots.get("red") or []
        blues = dots.get("blue") or []
        greens = dots.get("green") or []
        selfp = greens[0] if greens else None
        red_d = min([math.hypot(p[0] - selfp[0], p[1] - selfp[1]) for p in reds], default=9.9) \
            if selfp else 9.9
        feats.append({"t": s["t"], "hp": hp,
                      "dark": {k: float(hsv.get(k, 1.0)) for k in ("1", "2", "3")},
                      "e_d": e_d, "red_d": red_d, "n_red": len(reds), "n_blue": len(blues)})

    # 按下检测: dark 从 <0.55 跳到 >=0.75 (冷却开始=按下后)
    for k in ("1", "2", "3"):
        presses = []
        for i in range(1, len(feats)):
            d0 = feats[i - 1]["dark"].get(k, 1.0)
            d1 = feats[i]["dark"].get(k, 1.0)
            if d0 < 0.55 and d1 >= 0.75:
                presses.append(i)
        print(f"\n=== 技能{k} 按下 {len(presses)} 次 ===")
        if not presses:
            continue
        dists = collections.Counter()
        hp_ = collections.Counter()
        redd = collections.Counter()
        nd = collections.Counter()
        for i in presses:
            f = feats[i]
            ed = f["e_d"]
            db = "屏敌<0.25" if ed < 0.25 else ("屏敌0.25-0.45" if ed < 0.45 else (
                "屏敌0.45-0.8" if ed < 0.8 else "屏敌远"))
            dists[db] += 1
            hp_["HP<40%" if f["hp"] < 0.4 else ("HP40-80%" if f["hp"] < 0.8 else "HP>80%")] += 1
            rd = f["red_d"]
            redd["红近<0.35" if rd < 0.35 else ("红中<0.6" if rd < 0.6 else "红远")] += 1
            nd["红点" + str(f["n_red"])] = nd.get("红点" + str(f["n_red"]), 0) + 1
        for name, cnt in (("屏幕敌距", dists), ("HP", hp_), ("小地图红点", redd)):
            print(f"  {name}: " + ", ".join(f"{a}={b}" for a, b in cnt.most_common()))


if __name__ == "__main__":
    main()
