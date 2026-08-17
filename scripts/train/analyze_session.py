# -*- coding: utf-8 -*-
"""会话动作统计：分析 Agent 会话的决策分布与节奏（真机实测复盘用）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\analyze_session.py data\\matches\\<会话目录>
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) < 2:
        print("用法: analyze_session.py <会话目录>")
        return 1
    sess = Path(sys.argv[1])
    acts = [json.loads(l) for l in (sess / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    states = [json.loads(l) for l in (sess / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    print(f"会话: {sess.name}")
    print(f"状态: {len(states)} | 动作: {len(acts)}")

    # 动作类型分布
    kinds = Counter(a.get("type") for a in acts)
    print("动作类型:", dict(kinds))
    reasons = Counter(a.get("reason", "?") for a in acts)
    print("决策原因:", dict(reasons.most_common(12)))

    # 技能明细
    skills = [a for a in acts if a.get("type") == "skill"]
    print(f"\n技能: {len(skills)} 次（id 分布 {dict(Counter(a.get('id') for a in skills))}）")

    # 时间线（分钟粒度）
    if states:
        t0 = states[0]["t"]
        per_min = Counter(int((a.get("t", t0) - t0) // 60) for a in acts)
        print("分钟动作数:", dict(sorted(per_min.items())))

    # 钩子射程数据
    hook_file = ROOT / "data" / "measure_hook.jsonl"
    if hook_file.exists():
        recs = [json.loads(l) for l in hook_file.read_text(encoding="utf-8").splitlines()]
        hits = [r for r in recs if r.get("hit")]
        print(f"\n钩子射程累计: {len(recs)} 次（勾中 {len(hits)}）")
        if hits:
            ds = [r["dist_frac"] for r in hits]
            print(f"  勾中距离: 最大 {max(ds):.3f} 均值 {sum(ds)/len(ds):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
