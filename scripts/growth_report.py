# -*- coding: utf-8 -*-
"""每把自我学习反馈网页生成器 (v12.0)。

对局结束后调用 (POST_MATCH/自动学习后):
  读取: data/selfplay/*.jsonl (最近局), configs/evolved_policy.json (进化策略),
        configs/action_value.json (标定收益), docs/LEARNING_LOG.md (事件)
  生成: docs/selfplay_growth.html (网页, 越详细越好: 战绩/得分/进化/策略/数据/操作分布)

用法:
  python scripts/growth_report.py --session <file>   (生成某局报告)
  python scripts/growth_report.py --latest           (最新局)
  输出: docs/selfplay_growth.html (自动更新, 页面展示每次学习成长)
"""
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "selfplay"
OUT = ROOT / "docs" / "selfplay_growth.html"
CLS_NAME = {0: "敌英", 1: "队友", 8: "野怪", 11: "自己", 12: "小地图红点",
            13: "小地图蓝点", 14: "小地图绿点"}


def _load_records(session_path):
    recs = []
    with open(session_path, encoding="utf-8") as f:
        for ln in f:
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass
    return recs


def analyze(session_path):
    recs = _load_records(session_path)
    if not recs:
        return None
    total = float(recs[-1].get("total", 0.0))
    events = Counter()
    by_reason = Counter()
    score_by_reason = defaultdict(list)
    hp_series = []
    died_cnt = 0
    for r in recs:
        evt = r.get("event")
        if evt and evt.get("event"):
            events[evt["event"]] += 1
        by_reason[r.get("reason", "?")] += 1
        if r.get("dead"):
            died_cnt += 1
        if r.get("hp") is not None:
            hp_series.append(r["hp"])
        d = r.get("delta") or 0.0
        score_by_reason[r.get("reason", "?")].append(d)
    # high-performing reasons
    reason_avg = {k: round(sum(v) / len(v), 3) for k, v in score_by_reason.items() if len(v) >= 3}
    top_reasons = sorted(reason_avg.items(), key=lambda kv: -kv[1])[:8]
    bad_reasons = sorted(reason_avg.items(), key=lambda kv: kv[1])[:5]
    return {
        "session": Path(session_path).stem, "recs": len(recs), "total": round(total, 1),
        "events": dict(events), "by_reason": dict(by_reason.most_common(12)),
        "reason_avg": reason_avg, "top": top_reasons, "bad": bad_reasons,
        "died_ratio": round(died_cnt / max(1, len(recs)), 3),
        "hp_min": round(min(hp_series), 2) if hp_series else None,
    }


def evolved_policy_summary():
    try:
        p = json.loads((ROOT / "configs" / "evolved_policy.json").read_text(encoding="utf-8"))
        best = p.get("best_reason_by_state", {})
        return p.get("n", 0), best
    except Exception:
        return 0, {}


def html_report(all_reports, ep_n, ep_best):
    rows = ""
    for rep in all_reports:
        if not rep:
            continue
        ev = "、".join(f"{k}:{v}" for k, v in rep["events"].items()) or "无"
        top = "、".join(f"{r}({v:+.2f})" for r, v in rep["top"]) or "无"
        bad = "、".join(f"{r}({v:+.2f})" for r, v in rep["bad"]) or "无"
        rows += f"""
        <tr><td>{rep['session']}</td><td>{rep['recs']}</td><td>{rep['total']}</td>
        <td>{ev}</td><td>{rep['died_ratio']*100:.1f}%</td><td>{top}</td><td>{bad}</td></tr>"""
    # 进化策略表
    ep_rows = ""
    for sk, best in sorted(ep_best.items()):
        ep_rows += f"<tr><td>{sk}</td><td>{best}</td></tr>"
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI 学习成长报告</title>
<style>body{{font-family:'Microsoft YaHei';margin:20px;background:#111;color:#eee}}
h1{{color:#4af}}h2{{color:#fa4;border-bottom:1px solid #444;padding-bottom:5px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #444;padding:8px;font-size:13px}}
th{{background:#222}}tr:nth-child(even){{background:#1a1a1a}}tr:hover{{background:#222}}</style></head>
<body><h1>🎮 AI 王者荣耀 学习成长报告</h1>
<p>每把自动学习后更新 | 生成时间: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}</p>
<h2>📊 累计学习数据</h2>
<p>自我进化样本(状态-操作-得分记录): <b>{ep_n}</b> 条 | 详细见 data/selfplay/*.jsonl</p>
<h2>🧠 学到的进化策略 (状态 → 最佳操作)</h2>
<table><tr><th>状态</th><th>最佳操作</th></tr>{ep_rows}</table>
<h2>📈 各局学习情况</h2>
<table><tr><th>对局</th><th>记录数</th><th>总分</th><th>事件</th><th>死亡占比</th><th>高分操作</th><th>负分操作</th></tr>
{rows}</table>
<h2>💡 说明</h2>
<p>每把结束自动: 采集(状态+操作+得分) → evolve_policy 学"哪状态哪种操作得分高" → 更新策略表 → 下一局生效。</p>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"已生成: {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--session", metavar="FILE")
    args = ap.parse_args()
    if args.session:
        sess = [args.session]
    elif args.latest:
        fs = sorted(SP.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
        sess = [str(fs[-1])] if fs else []
    else:
        sess = [str(f) for f in sorted(SP.glob("*.jsonl"))]
    reports = []
    for s in sess:
        rp = analyze(s)
        if rp:
            reports.append(rp)
            print(f"分析 {rp['session']}: {rp['recs']} 条, 总分 {rp['total']}")
    ep_n, ep_best = evolved_policy_summary()
    html_report(reports, ep_n, ep_best)


if __name__ == "__main__":
    main()
