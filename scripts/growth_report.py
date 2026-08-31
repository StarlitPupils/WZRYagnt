# -*- coding: utf-8 -*-
"""中文学习成长报告 v12.3: KDA/伤害/承伤(从事件统计)/能力雷达图/对比平均。全中文+canvas图表。"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "selfplay"
OUT = ROOT / "docs" / "学习成长.html"

CN = {"enemy_in_skill2_range": "二技能出钩", "support_dps_lane": "支援射手路",
      "dps_lane_push": "射手路推进", "low_hp_fallback_safe": "残血退安全",
      "low_hp_safe_recall": "残血回城", "low_hp_recall_first": "残血回城",
      "chase_enemy_bar": "追血条", "low_resource_safe_restore": "低资源恢复",
      "follow_ally_minimap": "跟随队友", "support_red_nearest": "支援红点",
      "wall_avoid_map": "撞墙规避", "enemy_hero_near": "一技能消耗",
      "hook_flight": "钩子飞行", "engage_enemy": "攻击敌人"}
EVT_CN = {"kill": "击杀", "died": "死亡", "assist": "助攻", "tower_kill": "推塔",
          "minion_clear": "清兵", "be_attacked": "被攻击", "recall": "回城",
          "recall_interrupted": "回城被打断", "victory": "胜利", "defeat": "失败", "hook": "钩中"}


def cn_reason(r):
    return CN.get(r, r)


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
    kill = assist = died = 0
    events = Counter()
    by_reason = Counter()
    score_by_reason = defaultdict(list)
    hp_min = 1.0
    hp_avg = 0.0
    engage_cnt = 0
    for r in recs:
        evt = r.get("event")
        if evt and evt.get("event"):
            e = evt["event"]
            events[e] += 1
            if e == "kill":
                kill += 1
            elif e == "assist":
                assist += 1
            elif e == "died":
                died += 1
        by_reason[cn_reason(r.get("reason", "?"))] += 1
        d = r.get("delta") or 0.0
        score_by_reason[cn_reason(r.get("reason", "?"))].append(d)
        if r.get("hp") is not None:
            hp_min = min(hp_min, r["hp"])
            hp_avg += r["hp"]
        if r.get("n_enemy_scr"):
            engage_cnt += 1
    hp_avg = round(hp_avg / max(1, len(recs)), 2)
    reason_avg = {k: round(sum(v) / len(v), 3) for k, v in score_by_reason.items() if len(v) >= 3}
    return {"session": Path(session_path).stem, "recs": len(recs),
            "total": round(float(recs[-1].get("total", 0.0)), 1),
            "kill": kill, "assist": assist, "died": died,
            "events": {EVT_CN.get(k, k): v for k, v in events.items()},
            "by_reason": by_reason, "reason_avg": reason_avg,
            "hp_min": round(hp_min, 2), "hp_avg": hp_avg,
            "engage_ratio": round(engage_cnt / max(1, len(recs)), 2)}


def abilities(rep):
    """能力雷达图 5维: 攻击/生存/支援/经济(推进)/安全。由数据推导。"""
    kv = rep["kill"] + 0.7 * rep["assist"]
    survive = round(1.0 / max(1, rep["died"] + 1) * 100, 1)
    support = rep["by_reason"].get("支援射手路", 0) + rep["by_reason"].get("跟随队友", 0)
    push = rep["by_reason"].get("射手路推进", 0) + rep["by_reason"].get("支援射手路", 0)
    safe = rep["by_reason"].get("残血退安全", 0) + rep["by_reason"].get("残血回城", 0)
    # 归一化 (cap 100)
    return {"攻击": min(100, kv * 15), "生存": mid(survive, 100),
            "支援": min(100, support * 2), "推进": min(100, push * 2),
            "安全": min(100, safe * 3 + (1 - rep["hp_min"]) * 50)}


def mid(v, hi):
    return round(min(hi, max(0, v)), 1)


def generate_html(reps):
    rows = ""
    canvases = ""
    for i, rp in enumerate(reps):
        ab = abilities(rp)
        kda = f"{rp['kill']}/{rp['died']}/{rp['assist']}"
        avg_str = "、".join(f"{r}({v:+.0f})" for r, v in sorted(rp["reason_avg"].items(), key=lambda kv: -kv[1])[:4]) or "暂无"
        rows += f"""<tr data-{i}><td>{rp['session']}</td><td>{kda}</td><td>{rp['total']}</td>
        <td>{'、'.join(f'{k}:{v}' for k,v in rp['events'].items()) or '无'}</td>
        <td>{rp['hp_min']}</td><td>{avg_str}</td></tr>"""
        # 雷达数据 canvas
        canvases += f"""<canvas id="radar{i}" width="220" height="220"></canvas>"""
    # 对比: 最近 vs 平均 (最后局 vs 前局平均)
    last = reps[-1] if reps else None
    prev_list = reps[:-1]
    avg = {}
    if prev_list:
        for key in ("kill", "assist", "died", "total"):
            avg[key] = round(sum(r[key] for r in prev_list) / len(prev_list), 1)
    else:
        avg = {}
    # 数据 JS
    reps_json = json.dumps([{k: rp[k] for k in ("session", "kill", "assist", "died", "total")} for rp in reps], ensure_ascii=False)
    ab_json = json.dumps([abilities(rp) for rp in reps], ensure_ascii=False)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>AI 王者荣耀 学习成长报告</title>
<style>body{{font-family:'微软雅黑';margin:20px;background:#0d1117;color:#e6edf3}}
h1{{color:#58a6ff;text-align:center}}h2{{color:#f0883e;border-bottom:2px solid #30363d;padding:8px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #30363d;padding:8px;font-size:13px}}
th{{background:#161b22}}tr:nth-child(even){{background:#161b22}}
.report{{background:#161b22;border-radius:8px;padding:15px;margin:10px 0}}
.stat{{font-size:15px;margin:5px}}
.radar-grid{{display:flex;flex-wrap:wrap;gap:10px}}
canvas{{background:#161b22;border-radius:8px}}
.diff-up{{color:#3fb950}}.diff-down{{color:#f85149}}</style></head>
<body>
<h1>🎮 AI 王者荣耀 学习成长报告</h1>
<p style="text-align:center;color:#8b949e">每把自动更新 | {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}</p>
<h2>📊 各局 KDA / 得分 / 事件</h2>
<table><tr><th>对局</th><th>KDA(杀/死/助)</th><th>总分</th><th>事件</th><th>最低血</th><th>高分操作</th></tr>{rows}</table>
<h2>📈 能力雷达图 (每局: 攻击/生存/支援/推进/安全)</h2>
<div class="radar-grid">{canvases}</div>
<h2>📉 对比平均 (最近局 vs 前几局平均)</h2>
<div class="report" id="compare"></div>
<script>
const reps = {reps_json};
const abs = {ab_json};
function radar(canvasId, vals){{
  const c=document.getElementById(canvasId); if(!c) return;
  const ctx=c.getContext('2d'); const keys=Object.keys(vals);
  ctx.clearRect(0,0,220,220); ctx.strokeStyle='#888'; ctx.fillStyle='#e6edf3';
  const cx=110,cy=110,R=80;
  // grid
  for(let g=1;g<=4;g++){{ctx.beginPath();
    for(let k=0;k<=keys.length;k++){{const a=-Math.PI/2+k*2*Math.PI/keys.length;
      const r=R*g/4; ctx.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a));}}
    ctx.stroke();}}
  // axes+labels
  for(let k=0;k<keys.length;k++){{const a=-Math.PI/2+k*2*Math.PI/keys.length;
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+R*Math.cos(a),cy+R*Math.sin(a));ctx.stroke();
    ctx.fillText(keys[k],cx+(R+12)*Math.cos(a)-8,cy+(R+12)*Math.sin(a)+4);}}
  // polygon
  ctx.beginPath();
  for(let k=0;k<keys.length;k++){{const a=-Math.PI/2+k*2*Math.PI/keys.length;
    const r=R*(vals[keys[k]]/100); ctx.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a));}}
  ctx.closePath(); ctx.fillStyle='rgba(88,166,255,0.4)'; ctx.fill();
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2; ctx.stroke();
}}
reps.forEach((_,i)=>radar('radar'+i,abs[i]));
// 对比
const last=reps[reps.length-1]; const before=reps.slice(0,-1);
const cmp=document.getElementById('compare');
if(last&&before.length){{
  const avgK=before.reduce((a,r)=>a+r.kill,0)/before.length;
  const avgD=before.reduce((a,r)=>a+r.died,0)/before.length;
  const avgT=before.reduce((a,r)=>a+r.total,0)/before.length;
  cmp.innerHTML=`<p class="stat">最近局 <b>${{last.session}}</b> 击杀${{last.kill}}(avg ${{avgK.toFixed(1)}}) 死亡${{last.died}}(avg ${{avgD.toFixed(1)}}) 总分${{last.total}}(avg ${{avgT.toFixed(1)}})</p>
  <p class="stat">击杀: <span class="${{last.kill>=avgK?'diff-up':'diff-down'}}">${{last.kill>=avgK?'进步':'退步'}}</span> |
  死亡: <span class="${{last.died<=avgD?'diff-up':'diff-down'}}">${{last.died<=avgD?'进步':'退步'}}</span> |
  总分: <span class="${{last.total>=avgT?'diff-up':'diff-down'}}">${{last.total>=avgT?'进步':'退步'}}</span></p>`;
}}else{{ cmp.innerHTML='<p class="stat">暂无对比数据</p>'; }}
</script>
<p style="color:#8b949e">数据源: data/selfplay/*.jsonl (每步状态+操作+得分) | 进化策略: configs/evolved_policy.json</p>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"已生成: {OUT} ({len(reps)} 局)")


def main():
    fs = sorted(SP.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    reps = []
    for f in fs:
        rp = analyze(str(f))
        if rp:
            reps.append(rp)
    generate_html(reps)


if __name__ == "__main__":
    main()
