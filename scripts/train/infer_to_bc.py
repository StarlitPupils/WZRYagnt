# -*- coding: utf-8 -*-
"""反推动作 -> BC 数据集桥接：录像反推的动作流 + 会话状态 -> npz 训练集。

对齐：会话 states.jsonl 的 t 是 wall clock；反推 actions 的 t 是录像秒。
偏移关系（子代理实测得出）：video_t = wall_clock - OFFSET。
OFFSET 估计：录像时长 vs 会话时间跨度；或由 --offset 指定。
对齐后按 past-action 语义给每个状态配最近动作（与 states_to_dataset 一致）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\infer_to_bc.py
        --session data/matches/20260818_013610_484191
        --infer-actions temp/stream_actions_fixed.json
        --out data/datasets/bc_from_infer.npz [--offset 1786988141.44]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.train.encoding import encode_action, encode_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="MatchRecorder 会话目录")
    ap.add_argument("--infer-actions", required=True, help="infer_actions 输出 JSON")
    ap.add_argument("--out", required=True, help="输出 npz")
    ap.add_argument("--offset", type=float, default=None,
                    help="wall_clock - video_t 偏移；默认从录像时长与状态跨度估计")
    args = ap.parse_args()

    sess = Path(args.session)
    states = [json.loads(l) for l in (sess / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    infer = json.loads(open(args.infer_actions, encoding="utf-8").read())
    actions = [a for a in infer["actions"] if a.get("type") != "meta"]

    if args.offset is None:
        # 估计：录像帧数/帧率 = 录像秒；wall 跨度 = 状态跨度；假设录像覆盖会话
        fps = infer.get("fps") or 30.0
        vdur = infer.get("stats", {}).get("frames_processed", 0) / fps
        t0, t1 = states[0]["t"], states[-1]["t"]
        sspan = max(t1 - t0, 1e-3)
        # 录像开始对齐会话开始：offset = t0 - 0（录像 0s = 会话 t0 附近）
        offset = t0
        print(f"偏移估计: offset={offset:.2f}（录像 0s ↔ 会话 t0）")
    else:
        offset = args.offset

    feats, acts = [], []
    a_idx = 0
    a_times = [a.get("t", 0.0) + offset for a in actions]  # 转 wall clock
    for st in states:
        t = st.get("t", 0.0)
        while a_idx < len(a_times) and a_times[a_idx] <= t:
            a_idx += 1
        act = actions[a_idx - 1] if a_idx > 0 else {"type": "none"}
        # 动作格式转换 -> encode_action 兼容
        act_enc = {}
        if act["type"] == "move":
            act_enc = {"type": "move", "theta": _dir_to_theta(act.get("direction", "right")),
                       "r": 0.8}
        elif act["type"] == "skill":
            act_enc = {"type": "skill", "id": act.get("skill_id", 2), "mode": "tap"}
        elif act["type"] == "attack":
            act_enc = {"type": "attack", "priority": "free"}
        else:
            act_enc = {"type": "none"}
        feats.append(encode_state(st))
        acts.append(encode_action(act_enc))

    if not feats:
        print("无可编码状态")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        units=np.stack([f["units"] for f in feats]),
        unit_mask=np.stack([f["unit_mask"] for f in feats]),
        grid=np.stack([f["grid"] for f in feats]),
        ui=np.stack([f["ui"] for f in feats]),
        actions=np.stack(acts),
        metas=np.array([json.dumps(f["meta"], ensure_ascii=False) for f in feats]),
    )
    from collections import Counter
    dist = Counter(a["type"] for a in actions)
    print(f"数据集: {out}  {len(feats)} 样本")
    print(f"反推动作分布: {dict(dist)} | 对齐动作: move={(acts and int(sum(a[0] for a in acts)))}")
    return 0


def _dir_to_theta(direction: str) -> float:
    """反推 direction 标签 -> 弧度（0=右，逆时针；与执行器/编码一致）。"""
    import math
    m = {"right": 0.0, "right_up": math.pi / 4, "up": math.pi / 2,
         "left_up": 3 * math.pi / 4, "left": math.pi, "left_down": 5 * math.pi / 4,
         "down": 3 * math.pi / 2, "right_down": 7 * math.pi / 4}
    return m.get(direction, 0.0)


if __name__ == "__main__":
    sys.exit(main())
