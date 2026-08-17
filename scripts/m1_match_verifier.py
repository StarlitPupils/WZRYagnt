# -*- coding: utf-8 -*-
"""真局自动验证器：监听画面直到进入真实对局（状态机+小地图双确认），
然后自动执行 60 秒感知采样（11 类检测 + 小地图 + UI 数值）并输出验证报告。

用法（进对局前启动，进对局后自动工作）：
    venv\\Scripts\\python.exe scripts\\m1_match_verifier.py
        [--model runs/detect/zhongkui_11cls/weights/best.pt]
        [--window 60] [--hz 5]
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.calib import load_calibration  # noqa: E402
from wzry.capture.adb import AdbCapture  # noqa: E402
from wzry.data.collector import MatchRecorder  # noqa: E402
from wzry.state.match_state import MatchStateMachine  # noqa: E402
from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.minimap_tracker import MinimapTracker  # noqa: E402
from wzry.vision.ui_reader import read_ui  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/zhongkui_11cls/weights/best.pt")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=5.0)
    args = ap.parse_args()

    calib, _ = load_calibration()
    mc = calib.get("minimap_center", [0.086, 0.129])
    with open(ROOT / "configs" / "calibration_absolute.json", encoding="utf-8") as f:
        pts = json.load(f)["points"]
    skills = {1: pts["skill1"], 2: pts["skill2"], 3: pts["skill3"]}

    cap = AdbCapture()
    sm = MatchStateMachine(minimap_center_norm=mc)
    tracker = None
    confirm = 0
    print("监听中：请进入训练营/人机对局（检测到真实对局后自动采样 60 秒）...")

    while True:
        frame, _ = cap.get_frame()
        if frame is None:
            time.sleep(1.0)
            continue
        phase = sm.update(frame)
        if phase.value == "in_match":
            if tracker is None:
                h, w = frame.shape[:2]
                tracker = MinimapTracker(prior_center=[int(mc[0] * w), int(mc[1] * h)])
            mm = tracker.update(frame)
            if mm["found"]:
                confirm += 1
                if confirm >= 2:
                    break
            else:
                confirm = 0
        else:
            confirm = 0
        time.sleep(0.5)
    print("对局确认！开始 60 秒感知采样...")

    det = YoloDetector(args.model, conf=0.35)
    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")
    recorder.start(meta={"source": "match_verifier", "model": args.model})
    t_end = time.time() + args.window
    cls_counts = Counter()
    mm_blue, mm_red, mm_yellow = [], [], []
    hp_ratios, skill_dark = [], {1: [], 2: [], 3: []}
    n_frames = 0
    last = 0.0
    while time.time() < t_end:
        frame, _ = cap.get_frame()
        if frame is None:
            time.sleep(0.3)
            continue
        now = time.time()
        if now - last < 1.0 / args.hz:
            continue
        last = now
        n_frames += 1
        mm = tracker.update(frame)
        if mm["found"]:
            mm_blue.append(len(mm["dots"]["blue"]))
            mm_red.append(len(mm["dots"]["red"]))
            mm_yellow.append(len(mm["dots"]["yellow"]))
        dets = det.detect(frame)
        for d in dets:
            cls_counts[d.cls] += 1
        ui = read_ui(frame, skills)
        if ui["hp_bar"]:
            hp_ratios.append(ui["hp_bar"][-1])
        for k, v in ui["skills"].items():
            if v["dark_frac"] is not None:
                skill_dark[k].append(v["dark_frac"])
        # 真实会话采集（与验证并行）
        from wzry.state.fuser import build_state
        st = build_state(frame, dets, "in_match", minimap={
            "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
            "dots": mm["dots"], "towers": mm["towers"]}, frame_id=n_frames)
        st.ui = {"hp": ui["hp_bar"][-1] if ui["hp_bar"] else None,
                 "skill_dark": {k: v["dark_frac"] for k, v in ui["skills"].items()}}
        recorder.on_state(st.to_dict())
        recorder.on_frame(frame)
    recorder.close()

    def stat(name, lst):
        if lst:
            print(f"  {name:<16} 均值 {statistics.mean(lst):.3f}  样本 {len(lst)}")
        else:
            print(f"  {name:<16} 无样本")

    print(f"\n=== 真局验证报告（{n_frames} 帧）===")
    print("检测计数:", dict(cls_counts))
    print("小地图:")
    stat("蓝点", mm_blue)
    stat("红点", mm_red)
    stat("黄点", mm_yellow)
    print("UI:")
    stat("HP 比例", hp_ratios)
    for k in (1, 2, 3):
        stat(f"技能{k} dark_frac", skill_dark[k])
    out = ROOT / "temp" / "match_verify.json"
    out.write_text(json.dumps({
        "n_frames": n_frames, "cls_counts": dict(cls_counts),
        "mm_blue": {"mean": statistics.mean(mm_blue) if mm_blue else None, "n": len(mm_blue)},
        "mm_red": {"mean": statistics.mean(mm_red) if mm_red else None, "n": len(mm_red)},
        "hp": {"mean": statistics.mean(hp_ratios) if hp_ratios else None, "n": len(hp_ratios)},
        "skill_dark": {k: {"mean": statistics.mean(v) if v else None, "n": len(v)}
                       for k, v in skill_dark.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
