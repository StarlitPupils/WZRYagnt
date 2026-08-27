# -*- coding: utf-8 -*-
"""自我复盘器 (agent 局末自动运行):
  1) 本局事件统计: kill/assist/died/tower/被塔/被围
  2) 路线分布: 支援目标在下路走廊/中路/其他 (用户目标 45/45)
  3) 位置稳定性: 小地图点位帧间跳变率(闪烁指数)
  4) 输出 temp/reflect/<ts>.md 复盘报告 + configs/policy_tune.json 自调参建议
"""
import json
import math
import collections
import glob
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _strip_keys(d):
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
    return d


def reflect():
    # 最新行为审计(本局: 最后一段 in_match)
    beh = []
    try:
        for l in Path(ROOT / "temp" / "live_annot" / "behavior.jsonl").read_text(
                encoding="utf-8").splitlines():
            beh.append(json.loads(l))
    except Exception:
        pass
    if not beh:
        print("无行为审计数据")
        return
    # 本局: 取最后连续段(距最后 30 分钟内)
    t_end = beh[-1]["t"]
    seg = [r for r in beh if r["t"] >= t_end - 1500]
    ev = collections.Counter()
    route = collections.Counter()
    hp_min = 1.0
    for r in seg:
        reason = r.get("reason") or ""
        if "support" in reason:
            route["support"] += 1
        elif "chase" in reason:
            route["chase"] += 1
        elif "follow_ally" in reason:
            route["follow"] += 1
        # 奖励事件由 score 差分推: 分差>0 且带 kill/tower
        ev[reason] += 1
        hp_min = min(hp_min, r.get("hp", 1.0) if r.get("hp") is not None else 1.0)
    # 分数事件: 用 score jump 推断
    jumps = collections.Counter()
    prev = None
    for r in seg:
        s = r.get("score")
        if prev is not None and s is not None and prev is not None:
            d = s - prev
            if abs(d) >= 5:
                jumps[round(d)] += 1
        prev = s
    score_final = seg[-1].get("score", 0) if seg else 0

    # 位置闪烁: 从最近归档 states(小地图蓝/红/绿点逐帧跳变)
    flip = 0
    n = 0
    try:
        files = sorted(glob.glob(str(ROOT / "data" / "matches" / "20260826_*" / "states.jsonl")),
                       key=lambda p: Path(p).stat().st_mtime)
        if files:
            prev_dots = None
            for l in Path(files[-1]).read_text(encoding="utf-8").splitlines():
                d = json.loads(l)
                if d.get("phase") != "in_match":
                    continue
                mm = d.get("minimap") or {}
                dots = (mm.get("dots") or {}) if mm.get("found") else {}
                cur = (len(dots.get("blue", [])), len(dots.get("red", [])))
                if prev_dots is not None:
                    n += 1
                    # 计数突增减>2 = 疑似闪烁
                    if abs(cur[0] - prev_dots[0]) > 2 or abs(cur[1] - prev_dots[1]) > 2:
                        flip += 1
                prev_dots = cur
    except Exception:
        pass
    flip_rate = flip / max(1, n)

    # 报告
    out_dir = ROOT / "temp" / "reflect"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = [
        f"# 自我复盘 {ts}",
        f"## 最终得分: {score_final}  最低HP: {hp_min:.2f}",
        f"## 分数事件(奖励/惩罚): " + ", ".join(f"{k}分×{v}" for k, v in jumps.most_common(8)),
        f"## 行动分布: " + ", ".join(f"{k}={v}" for k, v in route.most_common()),
        f"## 位置闪烁率(点位突增减): {flip_rate*100:.0f}% ({flip}/{n})",
        f"## 反思建议:",
        "  - 若跳变率>20%: 继续强化 tracker 时序(当前已有中值滤波+跳变守卫)",
        "  - 若 died 次数>2: 支援血线+0.05 / 巡逻时长-",
        "  - 助攻+击杀=0: 进场深度不足, 提高缠斗距离/勾速",
    ]
    (out_dir / f"{ts}.md").write_text("\n".join(rep), encoding="utf-8")
    print(open(out_dir / f"{ts}.md", encoding="utf-8").read())

    # 自调参建议 (策略参数微调表)
    tune = {"flip_rate": round(flip_rate, 3), "score_final": score_final}
    if jumps.get(-50, 0) >= 2:
        tune["support_hp_penalty"] = 0.05
    if jumps.get(20, 0) + jumps.get(15, 0) == 0:
        tune["engage_frac_push"] = 0.05
    try:
        (ROOT / "configs" / "policy_tune.json").write_text(
            json.dumps(tune, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    reflect()
