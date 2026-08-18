# -*- coding: utf-8 -*-
"""触摸流手势解析：把 demo 会话的 touch.jsonl（getevent 原始事件）解析为"用户手势"，
并映射到逻辑屏幕坐标 + 匹配校准按钮，输出 gestures.jsonl。

坐标映射（MuMu SurfaceOrientation=1，横屏 1280x720）：
    逻辑 x = 物理 Y，逻辑 y = 720 - 物理 X

手势定义：
    {"t0": wall, "t1": wall, "dur": 秒, "kind": "tap"|"drag"|"hold",
     "x0","y0": 按下逻辑坐标, "x1","y1": 抬起逻辑坐标,
     "dx","dy": 位移, "dist": 位移距离(逻辑px), "theta": 位移方向(rad, 屏幕系),
     "button": 命中的校准按钮名 or null, "near": [按钮名, 距离px]}

用法：
    venv\\Scripts\\python.exe scripts\\train\\parse_touch.py data\\demos\\<session>
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 触摸事件代码（getevent hex）
EV_SYN = "0000"
EV_KEY = "0001"
EV_ABS = "0003"
BTN_TOUCH = "014a"
TRACKING_ID = "0039"
ABS_MT_SLOT = "002f"
ABS_X = "0035"
ABS_Y = "0036"


def parse_touch_file(path):
    """读取 touch.jsonl，按标准 MT 协议还原多指触摸手势。

    MuMu getevent MT 协议：
      002f ABS_MT_SLOT        -> 当前槽位（手指）
      0039 TRACKING_ID        -> 槽位手指 id；0xc350 表示该指抬起
      0035/0036 POSITION_X/Y  -> 更新当前槽位坐标
      014a BTN_TOUCH 1/0      -> 主触摸按下/抬起
      0000 SYN_REPORT         -> 一帧事件结束，提交坐标点

    每个槽位一个手指，手势 = 手指按下(TRACKING_ID 新值) 到 抬起(0xc350)。
    """
    gestures = []
    fingers = {}  # slot -> {"t0","id","pts":[(x,y),...]}
    slot = 0
    pending = {}  # slot -> [x, y] 本帧更新
    cur_x = {}
    cur_y = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o.get("kind") != "touch":
            continue
        t = float(o["t"])
        parts = o["raw"].split()
        if len(parts) < 3:
            continue
        typ, code, val = parts[0], parts[1], int(parts[2], 16)
        if typ == EV_ABS and code == ABS_MT_SLOT:
            slot = val
        elif typ == EV_ABS and code == TRACKING_ID:
            if val == 0xC350:
                # 该槽位手指抬起
                f = fingers.pop(slot, None)
                if f is not None:
                    f["t1"] = t
                    if f["pts"]:
                        gestures.append(f)
                    cur_x.pop(slot, None)
                    cur_y.pop(slot, None)
            elif slot not in fingers:
                # 新手指
                fingers[slot] = {"t0": t, "t1": t, "id": val, "pts": []}
        elif typ == EV_ABS and code == ABS_X:
            cur_x[slot] = val
        elif typ == EV_ABS and code == ABS_Y:
            cur_y[slot] = val
        elif typ == EV_ABS and code == TRACKING_ID:
            pass
        elif typ == EV_SYN and code == "0000":
            # SYN_REPORT：提交当前槽位坐标（所有活跃手指）
            for s, f in list(fingers.items()):
                if s in cur_x and s in cur_y:
                    f["pts"].append((cur_x[s], cur_y[s]))
                    f["t1"] = t
    # 未抬起的尾巴
    for f in fingers.values():
        if f["pts"]:
            gestures.append(f)
    return gestures


def phys_to_logical(px, py):
    """物理 (px∈[0,720], py∈[0,1280]) -> 逻辑 (x∈[0,1280], y∈[0,720])。"""
    x = py
    y = 720 - px
    return x, y


def build_gestures(gestures_raw):
    out = []
    for g in gestures_raw:
        pts = g.get("pts", [])
        if len(pts) < 1:
            continue
        x0, y0 = phys_to_logical(pts[0][0], pts[0][1])
        x1, y1 = phys_to_logical(pts[-1][0], pts[-1][1])
        dur = g["t1"] - g["t0"]
        dist = math.hypot(x1 - x0, y1 - y0)
        if dur <= 0.25 and dist < 30:
            kind = "tap"
        elif dist >= 30:
            kind = "drag"
        else:
            kind = "hold"
        theta = math.atan2(-(y1 - y0), x1 - x0) if dist > 5 else None
        out.append({
            "t0": round(g["t0"], 3), "t1": round(g["t1"], 3),
            "dur": round(dur, 3), "kind": kind,
            "x0": round(x0, 1), "y0": round(y0, 1),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "dx": round(x1 - x0, 1), "dy": round(y1 - y0, 1),
            "dist": round(dist, 1),
            "theta": round(theta, 3) if theta is not None else None,
        })
    return out


def load_calib_points():
    p = ROOT / "configs" / "calibration_absolute.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data["points"].items()}


def match_button(gs, points, radius=70.0):
    """手势起点命中哪个校准按钮。"""
    for g in gs:
        best, best_d = None, 1e9
        for name, (bx, by) in points.items():
            d = math.hypot(g["x0"] - bx, g["y0"] - by)
            if d < best_d:
                best, best_d = name, d
        g["near"] = [best, round(best_d, 1)]
        g["button"] = best if best_d <= radius else None
    return gs


def main():
    ap = argparse.ArgumentParser(description="触摸流手势解析")
    ap.add_argument("sess", help="会话目录 data/demos/<session>")
    ap.add_argument("--radius", type=float, default=70.0, help="按钮命中半径(px)")
    args = ap.parse_args()

    sess = Path(args.sess)
    touch_file = sess / "touch.jsonl"
    if not touch_file.exists() or touch_file.stat().st_size == 0:
        print("无触摸数据")
        return 0
    raw = parse_touch_file(touch_file)
    gs = build_gestures(raw)
    gs = match_button(gs, load_calib_points(), args.radius)
    (sess / "gestures.jsonl").write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in gs) + "\n",
        encoding="utf-8")
    print(f"手势 {len(gs)} 个（原始事件段 {len(raw)}）-> {sess / 'gestures.jsonl'}")

    from collections import Counter
    kinds = Counter(g["kind"] for g in gs)
    btns = Counter(g["button"] for g in gs if g["button"])
    print("手势类型:", dict(kinds))
    print("按钮命中:", dict(btns.most_common(15)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
