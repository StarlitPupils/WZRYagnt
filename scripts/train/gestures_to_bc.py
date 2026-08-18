# -*- coding: utf-8 -*-
"""示范局手势 -> BC 数据集桥接：用户真实操作（gestures.jsonl）+ 状态流 -> npz 训练集。

与 infer_to_bc 的区别：
  - 动作来源是【用户真实触摸/键盘手势】（parse_touch.py 产出），不是录像反推；
  - 对齐：gestures.t0/t1 与 states.t 同为 wall clock，直接按时间对齐，无需偏移；
  - 持续语义：drag（WASD 移动）从 t0 持续到 t1；tap（技能/攻击/回城）在 t0 时刻生效，
    延续 1 秒窗口（普攻/技能释放后的生效期），其后回 none。

用法：
    venv\\Scripts\\python.exe scripts\\train\\gestures_to_bc.py data\\demos\\<session>
        [--out data/datasets/bc_demo_<session>.npz]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.train.encoding import encode_state, encode_action  # noqa: E402

TAP_LINGER_S = 1.0   # tap 动作延续窗口（普攻间隔/技能生效）
BUTTON_TO_ACT = {
    "skill1": {"type": "skill", "id": 1},
    "skill2": {"type": "skill", "id": 2},
    "skill3": {"type": "skill", "id": 3},
    "attack": {"type": "attack"},
    "attack_minion": {"type": "attack"},
    "attack_tower": {"type": "attack"},
    "summoner": {"type": "summoner"},
    "recall": {"type": "recall"},
    "restore": {"type": "restore"},
}


def load_calib_pts():
    p = ROOT / "configs" / "calibration_absolute.json"
    return json.loads(p.read_text(encoding="utf-8"))["points"]


CALIB_PTS = load_calib_pts()
STICK_CX, STICK_CY = CALIB_PTS["move_stick_center"]


def dir_theta(btn):
    """dir_* 按钮相对摇杆中心的方位角（弧度，0=右，逆时针）。"""
    bx, by = CALIB_PTS[btn]
    return math.atan2(-(by - STICK_CY), bx - STICK_CX)


def gesture_to_events(g):
    """一个手势 -> 0..N 条动作事件（wall clock 时间轴）。
    返回 [(t0, t1, act_dict)]：动作在 [t0, t1] 生效。
    """
    kind = g.get("kind")
    btn = g.get("button")
    if kind == "drag":
        # 摇杆拖动 = 移动（WASD 按住或拖摇杆）：theta 已由 parse_touch 算好
        if g.get("theta") is None:
            return []
        return [(float(g["t0"]), float(g["t1"]),
                 {"type": "move", "theta": float(g["theta"]), "r": 1.0})]
    if kind == "tap":
        if btn and btn.startswith("dir_"):
            # WASD 短按 = 该方向移动指令（生效窗口用短按时长 + 移动响应）
            return [(float(g["t0"]), float(g["t1"]) + TAP_LINGER_S,
                     {"type": "move", "theta": dir_theta(btn), "r": 1.0})]
        if btn in BUTTON_TO_ACT:
            act = dict(BUTTON_TO_ACT[btn])
            t0 = float(g["t0"])
            return [(t0, t0 + TAP_LINGER_S, act)]
    return []


def build_action_stream(gestures):
    """所有手势 -> 排序后的生效动作区间列表。"""
    events = []
    for g in gestures:
        events.extend(gesture_to_events(g))
    events.sort(key=lambda e: e[0])
    return events


def action_at(t, events, idx):
    """past-action 语义：返回 t 时刻生效的动作（无则 none）。
    生效判定：最后一条 t0 <= t 的事件；若其区间覆盖 t（t1 >= t）则采用，
    否则视为"已结束"，回 none（避免把结束的移动延续成 none 错误标注）。
    """
    best = None
    while idx[0] < len(events) and events[idx[0]][0] <= t:
        e = events[idx[0]]
        if e[1] >= t:
            best = e[2]
        idx[0] += 1
    if best is not None:
        return best
    # 指针已越过但区间仍覆盖 t 的事件（drag 长移动）：从当前位置向前找
    i = idx[0] - 1
    while i >= 0 and events[i][1] >= t:
        if events[i][0] <= t:
            return events[i][2]
        i -= 1
    return {"type": "none"}


def main():
    ap = argparse.ArgumentParser(description="示范局手势 -> BC 数据集")
    ap.add_argument("sess", help="会话目录 data/demos/<session>")
    ap.add_argument("--out", default=None, help="输出 npz 路径")
    args = ap.parse_args()

    sess = Path(args.sess)
    states = [json.loads(l) for l in
              (sess / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    gestures = [json.loads(l) for l in
                (sess / "gestures.jsonl").read_text(encoding="utf-8").splitlines()]
    if not states or not gestures:
        print(f"数据不足: states={len(states)} gestures={len(gestures)}")
        return 1
    print(f"会话 {sess.name}: 状态 {len(states)} | 手势 {len(gestures)}")

    events = build_action_stream(gestures)
    print(f"动作区间 {len(events)} 条（drag 移动 + tap 技能/攻击/回城/恢复）")

    feats, acts = [], []
    idx = [0]
    dist = {}
    for st in states:
        t = st.get("t", 0.0)
        act = action_at(t, events, idx)
        t0, t1 = act.get("t0"), act.get("t1")
        if t0 is not None:
            act = {k: v for k, v in act.items() if k not in ("t0", "t1")}
        dist[act.get("type", "none")] = dist.get(act.get("type", "none"), 0) + 1
        feats.append(encode_state(st))
        acts.append(encode_action(act))

    out = Path(args.out) if args.out else (
        ROOT / "data" / "datasets" / f"bc_demo_{sess.name}.npz")
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
    print(f"数据集已写出: {out}  {len(feats)} 样本")
    print("动作标签分布:", dict(sorted(dist.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
