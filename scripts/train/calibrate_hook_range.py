# -*- coding: utf-8 -*-
"""钩子射程标定：分析 data/measure_hook.jsonl（Agent v2.1 自动记录的钩子释放距离与命中），
给出二技能范围阈值建议。

数据来源：Agent 在训练营对局中用 --action 运行，每次释放二技能记录
{"t", "dist_frac": 释放时最近敌人距离(屏宽比例), "hit": 是否勾中}。

输出：
  - 勾中距离分布（最大/均值）
  - 建议 SKILL2_RANGE_FRAC = 勾中最大距离 × 0.92（留余量）

用法：
    venv\\Scripts\\python.exe scripts\\train\\calibrate_hook_range.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = ROOT / "data" / "measure_hook.jsonl"
    if not p.exists():
        print(f"无测量数据: {p}（先让 Agent 用 --action 在训练营跑几局）")
        return 1
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    hits = [r for r in recs if r.get("hit")]
    misses = [r for r in recs if not r.get("hit")]
    print(f"记录总数: {len(recs)} | 勾中: {len(hits)} | 未中: {len(misses)}")
    if hits:
        ds = [r["dist_frac"] for r in hits]
        print(f"勾中距离: 最大 {max(ds):.3f} 均值 {sum(ds)/len(ds):.3f} "
              f"最小 {min(ds):.3f}（屏宽比例）")
        print(f"勾中距离明细: {sorted(round(d, 3) for d in ds)}")
        suggest = max(ds) * 0.92
        print(f"\n建议 SKILL2_RANGE_FRAC = {suggest:.3f}（勾中最大距离 × 0.92 留余量）")
    if misses:
        ds = [r["dist_frac"] for r in misses]
        print(f"\n未中距离: 最大 {max(ds):.3f}（>阈值部分属于正常脱靶；"
              f"若未中距离普遍低于勾中距离说明判定或勾法有问题）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
