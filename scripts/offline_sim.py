# -*- coding: utf-8 -*-
"""第二轮学习: 用户 states 回放 AI decide(), 对比行为分布 与 用户实际行为分布。"""
import sys
import json
import math
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "data" / "demos" / "20260818_125747"
from scripts.m2_agent_v2 import decide  # noqa: E402

# 与 learn_from_user 相同的标签器(简化)
def label_for(disp, hp, red_d, blue_d, ne_dist, dirv):
    dispn = math.hypot(*disp) or 1e-6
    if hp < 0.40 and red_d > 0.5:
        return "撤退"
    if red_d < 0.35 and hp >= 0.4:
        return "支援"
    if blue_d < 0.4 and dispn > 0.015:
        return "跟队"
    if red_d < 0.5 and dispn > 0.02:
        return "支援"
    return "跟队"

def main():
    states = []
    for ln in (DEMO / "states.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("phase") != "in_match":
            continue
        states.append(d)
    user_cnt = collections.Counter()
    ai_cnt = collections.Counter()
    agree = 0
    n = 0
    for i, s in enumerate(states):
        t = s["t"]
        ui = s.get("ui") or {}
        hp = float(ui.get("hp") or 1.0)
        mm = s.get("minimap") or {}
        dots = (mm.get("dots") or {}) if mm.get("found") else {}
        blues = dots.get("blue") or []
        reds = dots.get("red") or []
        greens = dots.get("green") or []
        selfp = greens[0] if greens else (blues[0] if blues else None)
        units = s.get("units") or []
        ne_dist = 9.9
        for u in units:
            if u.get("cls") in ("enemy_hero", "enemy_minion"):
                scr = u.get("screen") or [0.5, 0.5]
                ne_dist = min(ne_dist, math.hypot(scr[0] - 0.5, scr[1] - 0.5))
        if not selfp or not reds:
            continue
        # 用户标签: 未来3s蓝群位移
        j = i
        while j < len(states) and states[j]["t"] - t < 3.0:
            j += 1
        if j >= len(states):
            j = len(states) - 1
        d2 = (states[j].get("minimap") or {}).get("dots") or {}
        blues2 = d2.get("blue") or []
        if not blues2:
            continue
        c1 = [sum(p[0] for p in blues) / len(blues), sum(p[1] for p in blues) / len(blues)]
        c2 = [sum(p[0] for p in blues2) / len(blues2), sum(p[1] for p in blues2) / len(blues2)]
        disp = (c2[0] - c1[0], c2[1] - c1[1])
        red_d = min([math.hypot(p[0] - selfp[0], p[1] - selfp[1]) for p in reds], default=9.9)
        blue_d = min([math.hypot(p[0] - selfp[0], p[1] - selfp[1]) for p in blues], default=9.9)
        ulabel = label_for(disp, hp, red_d, blue_d, ne_dist, None)
        user_cnt[ulabel] += 1
        # AI 决策
        minmap_for_ai = {"found": True, "dots": {"blue": blues, "red": reds,
                                                 "green": greens, "yellow": dots.get("yellow", [])},
                         "towers": []}
        state_dict = {"t": t, "screen_size": [1280, 720],
                      "units": [{"cls": u.get("cls"),
                                 "screen": u.get("screen", [0.5, 0.5])} for u in units],
                      "minimap": minmap_for_ai, "ui": {"hp": hp, "mp": 1.0},
                      "extra": {}}
        try:
            act = decide(state_dict, {"camp": "blue", "match_start_t": t - 3000})
        except Exception:
            continue
        atype = act.get("type")
        if atype == "move":
            alabel = "支援" if red_d < 0.5 else "跟队"
            if hp < 0.40 and red_d > 0.5:
                alabel = "撤退"
        elif atype in ("skill", "attack", "summoner"):
            alabel = "战斗"
        else:
            alabel = "待机"
        ai_cnt[alabel] += 1
        agree += int(alabel == ulabel)
        n += 1
    print("样本:", n)
    print("\n=== 用户行为分布 vs AI(回放)行为分布 ===")
    for k in sorted(set(user_cnt) | set(ai_cnt)):
        u = user_cnt.get(k, 0); a = ai_cnt.get(k, 0)
        print(f"  {k}: 用户 {u} ({u/n*100:.0f}%) | AI {a} ({a/n*100:.0f}%)")
    print(f"\n一致率: {agree}/{n} = {agree/n*100:.0f}%")

if __name__ == "__main__":
    main()
