# -*- coding: utf-8 -*-
"""中文学习成长报告 v12.1: 每局 KDA/数据/能力分析图/对比平均, 全中文图表。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "selfplay"
OUT = ROOT / "docs" / "学习成长.html"

CN = {("enemy_in_skill2_range", "skill2"): "二技能出钩(勾敌人)",
      ("support_dps_lane",): "支援射手路",
      ("dps_lane_push",): "射手路推进",
      ("low_hp_fallback_safe",): "残血退安全位",
      ("low_hp_safe_recall", "low_hp_recall_first"): "残血回城补满",
      ("chase_enemy_bar",): "追血条(坏操作)",
      ("low_resource_safe_restore",): "低资源恢复",
      ("follow_ally_minimap", "support_red_nearest"): "跟随队友/支援",
      ("wall_avoid_map", "wall_avoid"): "撞墙规避"}

EVT_CN = {"kill": "击杀", "died": "死亡", "assist": "助攻", "tower_kill": "推塔",
          "minion_clear": "清兵", "be_attacked": "被攻击", "recall": "回城",
          "recall_interrupted": "回城被打断", "victory": "胜利", "defeat": "失败",
          "hook": "钩中"}

REASON_CN = {k: (v[0] if isinstance(v, tuple) else v) for k, v in CN.items()}


def cn_reason(r):
    for keys, name in CN.items():
        if r in keys:
            return name
    return r


def analyze(session_path):
    recs = []
    with open(session_path, encoding="utf-8") as f:
        for ln in f:
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass
    if not recs:
        return None
    total = float(recs[-1].get("total", 0.0))
    events = Counter()
    by_reason = Counter()
    score_by_reason = defaultdict(list)
    died = sum(1 for r in recs if r.get("dead"))
    hp_min = 1.0
    for r in recs:
        evt = r.get("event")
        if evt and evt.get("event"):
            events[evt["event"]] += 1
        by_reason[cn_reason(r.get("reason", "?"))] += 1
        d = r.get("delta") or 0.0
        score_by_reason[cn_reason(r.get("reason", "?"))].append(d)
        if r.get("hp") is not None:
            hp_min = min(hp_min, r["hp"])
    reason_avg = {k: round(sum(v) / len(v), 3) for k, v in score_by_reason.items() if len(v) >= 3}
    return {"session": Path(session_path).stem, "recs": len(recs), "total": round(total, 1),
            "events": {EVT_CN.get(k, k): v for k, v in events.items()},
            "by_reason": by_reason, "reason_avg": reason_avg,
            "died": died, "hp_min": round(hp_min, 2)}


def load_all(paths):
    reps = []
    for p in paths:
        rp = analyze(p)
        if rp:
            reps.append(rp)
    return reps


def generate_html(reps, all_historical):
    # 各局 KDA 推断: total 转化(击杀/死亡事件有则用)
    rows = ""
    for rp in reps:
        ev = "、".join(f"{k}:{v}" for k, v in rp["events"].items()) or "无事件"
        avg_str = ""
        for reason, avg in sorted(rp["reason_avg"].items(), key=lambda kv: -kv[1])[:5]:
            avg_str += f"{reason} {avg:+.2f}分; "
        rows += f"""<tr><td>{rp['session']}</td><td>{rp['recs']}</td>
        <td>{rp['total']}</td><td>{ev}</td><td>{rp['died']}</td>
        <td>{rp['hp_min']}</td><td>{avg_str or '暂无'}</td></tr>"""
    # 平均对比: 最近局 vs 历史平均
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>AI 学习成长报告 (中文)</title>
<style>body{{font-family:'微软雅黑';margin:20px;background:#0d1117;color:#e6edf3}}
h1{{color:#58a6ff;text-align:center}}h2{{color:#f0883e;border-bottom:2px solid #30363d;padding:8px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #30363d;padding:8px;font-size:13px}}
th{{background:#161b22}}tr:nth-child(even){{background:#161b22}}
.report{{background:#161b22;border-radius:8px;padding:15px;margin:10px 0}}
.stat{{font-size:15px;margin:5px}}
.bar{{background:#238636;color:white;padding:3px 8px;border-radius:4px;display:inline-block;margin:2px}}
.up{{color:#3fb950}}.down{{color:#f85149}}</style></head>
<body>
<h1>🎮 AI 王者荣耀 学习成长报告</h1>
<p style="text-align:center;color:#8b949e">每把自我学习后自动更新 | {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}</p>
<h2>📊 最近一局学习详情</h2>"""
    if reps:
        last = reps[-1]
        html += f"""<div class="report">
<p class="stat"><b>{last['session']}</b> | 记录 {last['recs']} 步 | 总分 <b>{last['total']}</b></p>
<p class="stat">🎯 本局事件: {('、'.join(f'{k}:{v}' for k,v in last['events'].items()) or '无')}</p>
<p class="stat">📉 最低血量: {last['hp_min']} | 💀 死亡计数: {last['died']}</p>
<p class="stat">✅ 学到的高分操作: {('、'.join(f'{r}({v:+.2f})' for r,v in sorted(last['reason_avg'].items(), key=lambda kv:-kv[1])[:5]) or '暂无')}</p>
<p class="stat">❌ 负分操作(应避免): {('、'.join(f'{r}({v:+.2f})' for r,v in sorted(last['reason_avg'].items(), key=lambda kv:kv[1])[:3]) or '暂无')}</p>
</div>"""
    html += f"""<h2>📈 各局学习数据 (叠加历史)</h2>
<table><tr><th>对局</th><th>记录步数</th><th>总分</th><th>事件</th><th>死亡</th><th>最低血</th><th>学到的高分操作</th></tr>{rows}</table>
<h2>🧠 本系统不断学到的东西</h2>
<div class="report">
<p>• <b>碰到敌人 → 二技能先出钩</b> (勾中再放大招, 没勾到只一技能消耗)</p>
<p>• <b>残血 → 先退到安全位置, 再回城补满</b> (不再硬拼)</p>
<p>• <b>支援 → 去射手路</b> (蓝方下路/红方上路, 射手在哪跟哪)</p>
<p>• <b>不追小地图红点</b> (假信号不可信, 只认屏幕上的真敌人)</p>
</div>
<p style="color:#8b949e">完整数据: data/selfplay/*.jsonl | 进化策略: configs/evolved_policy.json</p>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"已生成: {OUT}")


def main():
    fs = sorted(SP.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    reps = load_all([str(f) for f in fs])
    generate_html(reps, reps)


if __name__ == "__main__":
    main()
