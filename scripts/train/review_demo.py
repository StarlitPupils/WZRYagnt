# -*- coding: utf-8 -*-
"""示范局复盘时间线：把用户手动操作的一局（录屏反推动作 + GameState 流 + 触摸流）
对齐成逐段时间线，供用户逐段讲解决策，Agent 记录规则。

输入（data/demos/<session>/）：
    manifest.json   t0 = 录像起点 wall clock（对齐基准）
    actions.json    反推用户动作（t = 录像秒 -> wall = t0 + t）
    states.jsonl    GameState 流（t = wall clock）
    touch.jsonl     触摸事件（t = wall clock，物理坐标；可选）

输出：
    <session>/timeline.md   逐段时间线（默认每 5 秒一段）
    <session>/timeline.json 原始对齐数据

用法：
    venv\\Scripts\\python.exe scripts\\train\\review_demo.py data\\demos\\20260818_xxxxxx
    venv\\Scripts\\python.exe scripts\\train\\review_demo.py data\\demos\\xxxxx --step 10
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_jsonl(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def seg_range(t, step):
    return int(t // step) * step


def fmt_t(wall_t, t0):
    dt = wall_t - t0
    m, s = divmod(max(0, int(dt)), 60)
    return f"{m:02d}:{s:02d}"


def summarize_state(st):
    """把一条 GameState 压成一行摘要（只保留与决策相关的字段）。"""
    units = st.get("units") or []
    near = {"enemy_hero": [], "enemy_minion": [], "enemy_turret": [],
            "ally_hero": [], "ally_minion": []}
    for u in units:
        cls = u.get("cls", "")
        if cls in near:
            near[cls].append(u.get("screen") or [0, 0, 0, 0])
    ui = st.get("ui") or {}
    hp = ui.get("hp")
    dark = ui.get("skill_dark") or {}
    mm = st.get("minimap") or {}
    dots = mm.get("dots") or {}
    parts = []
    if hp is not None:
        parts.append(f"HP{hp:.2f}")
    for cls, label in (("enemy_hero", "敌英"), ("enemy_minion", "敌兵"),
                       ("enemy_turret", "敌塔"), ("ally_hero", "友英")):
        if near[cls]:
            d = min(math.hypot(s[0] - 0.5, (s[1] - 0.5) * 0.5625)
                    for s in near[cls])
            parts.append(f"{label}{len(near[cls])}@{d:.2f}")
    if mm.get("found"):
        parts.append(f"小地图蓝{len(dots.get('blue', []))}/红{len(dots.get('red', []))}")
    if dark:
        dk = {k: round(v, 2) for k, v in dark.items() if v > 0.15}
        if dk:
            parts.append(f"技能暗{dk}")
    return " ".join(parts) if parts else "（空）"


def fmt_action(a):
    t = a.get("type")
    if t == "move":
        th = a.get("direction")
        if isinstance(th, (list, tuple)) and len(th) == 2:
            return f"移动→({th[0]:.2f},{th[1]:.2f})"
        return f"移动θ={a.get('theta', '?')}"
    if t == "skill":
        return f"技能{a.get('skill_id', a.get('id', '?'))}"
    if t == "attack":
        return f"普攻({a.get('priority', 'free')})"
    if t == "recall":
        return "回城"
    if t == "summoner":
        return "召唤师"
    return str(a)


def main():
    ap = argparse.ArgumentParser(description="示范局复盘时间线")
    ap.add_argument("sess", help="会话目录 data/demos/<session>")
    ap.add_argument("--step", type=float, default=5.0, help="时间线段长（秒）")
    args = ap.parse_args()

    sess = Path(args.sess)
    manifest = json.loads((sess / "manifest.json").read_text(encoding="utf-8"))
    t0 = float(manifest.get("t0") or 0.0)
    states = load_jsonl(sess / "states.jsonl")
    acts = []
    if (sess / "actions.json").exists():
        data = json.loads((sess / "actions.json").read_text(encoding="utf-8"))
        acts = [a for a in data.get("actions", []) if a.get("type") != "meta"]
    touches = load_jsonl(sess / "touch.jsonl")

    print(f"会话 {sess.name} | t0={t0:.3f} | 状态 {len(states)} | "
          f"动作 {len(acts)} | 触摸 {len(touches)} 条")

    # ---- 动作转 wall clock ----
    acts_wall = []
    for a in acts:
        vt = a.get("t")
        if vt is None:
            continue
        a2 = dict(a)
        a2["wall"] = t0 + float(vt)
        a2["vt"] = float(vt)
        acts_wall.append(a2)
    acts_wall.sort(key=lambda a: a["wall"])

    # ---- 触摸/手势（gestures.jsonl 优先，其次 raw touch.jsonl）----
    gestures = load_jsonl(sess / "gestures.jsonl")
    touches_wall = []
    if gestures:
        for g in gestures:
            touches_wall.append({
                "t": float(g["t0"]), "kind": g["kind"], "dur": g.get("dur"),
                "button": g.get("button"), "near": g.get("near"),
                "x0": g.get("x0"), "y0": g.get("y0"),
                "theta": g.get("theta"), "dist": g.get("dist"),
            })
    else:
        touches_wall = [dict(t) for t in load_jsonl(sess / "touch.jsonl")]

    # ---- 状态按 wall 排序 ----
    states_sorted = sorted(states, key=lambda s: float(s.get("t") or 0.0))

    # ---- 分段聚合 ----
    t_start = states_sorted[0]["t"] if states_sorted else t0
    t_end = states_sorted[-1]["t"] if states_sorted else (t0 + 60)
    n_seg = int((t_end - t_start) // args.step) + 1
    segs = []
    for i in range(n_seg):
        s0 = t_start + i * args.step
        s1 = s0 + args.step
        seg_states = [s for s in states_sorted if s0 <= float(s.get("t") or 0) < s1]
        seg_acts = [a for a in acts_wall if s0 <= a["wall"] < s1]
        seg_touch = [t for t in touches_wall if s0 <= float(t.get("t") or 0) < s1]
        segs.append({
            "i": i, "s0": s0, "s1": s1,
            "label": f"{fmt_t(s0, t0)}-{fmt_t(s1, t0)}",
            "n_states": len(seg_states),
            "states": seg_states, "acts": seg_acts, "touches": seg_touch,
        })

    # ---- 输出 markdown ----
    lines = [f"# 示范局复盘时间线：{sess.name}",
             f"（t0={t0:.3f}，步长 {args.step}s，共 {len(segs)} 段）", ""]
    for seg in segs:
        if not seg["states"] and not seg["acts"]:
            continue
        lines.append(f"## [{seg['label']}]")
        # 状态摘要：取段内代表帧（首/中/尾各一）
        n = len(seg["states"])
        picks = [seg["states"][0]]
        if n > 2:
            picks.append(seg["states"][n // 2])
        if n > 1:
            picks.append(seg["states"][-1])
        for s in picks:
            lines.append(f"  - 状态 {fmt_t(s.get('t', 0), t0)}: {summarize_state(s)}")
        if seg["acts"]:
            ac = ", ".join(fmt_action(a) for a in seg["acts"])
            lines.append(f"  - 动作({len(seg['acts'])}): {ac}")
        if seg["touches"]:
            btn = {}
            for t in seg["touches"]:
                b = t.get("button") or t.get("raw", "")[:24]
                k = t.get("kind", "?")
                btn[f"{k}:{b}"] = btn.get(f"{k}:{b}", 0) + 1
            detail = "; ".join(f"{k}×{c}" for k, c in sorted(
                btn.items(), key=lambda kv: -kv[1])[:6])
            lines.append(f"  - 操作({len(seg['touches'])}): {detail}")
        lines.append("")
    md = "\n".join(lines)
    (sess / "timeline.md").write_text(md, encoding="utf-8")
    (sess / "timeline.json").write_text(
        json.dumps({
            "session": sess.name, "t0": t0, "step": args.step, "segs": segs,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"\n时间线已写入 {sess / 'timeline.md'}")
    # 快速统计
    kind = {}
    for a in acts_wall:
        kind[a.get("type")] = kind.get(a.get("type"), 0) + 1
    print(f"用户动作分布: {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
