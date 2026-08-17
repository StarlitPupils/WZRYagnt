# -*- coding: utf-8 -*-
"""M1 实时感知管线（骨架）：scrcpy 流式采集 + YOLO 检测 + 状态机 + GameState 融合。

用法：
    venv\\Scripts\\python.exe scripts\\m1_live_pipeline.py [--seconds 15] [--save-every 2.0]
        [--no-save] [--show] [--detect-hz 8]

说明：
  - 采集：scrcpy mkv 流（~30fps, ~33ms/帧），优于 screencap（167ms）；
  - 检测：best.pt（当前 4 类）按 --detect-hz 频率跑在最新帧上；
  - 状态机：识别是否对局中；对局中把 GameState 定期存档到 data/live_states/。
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.calib import load_calibration  # noqa: E402
from wzry.capture.scrcpy_stream import ScrcpyStreamCapture  # noqa: E402
from wzry.data.collector import MatchRecorder  # noqa: E402
from wzry.state.fuser import build_state  # noqa: E402
from wzry.state.match_state import MatchPhase, MatchStateMachine  # noqa: E402
from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.minimap_tracker import MinimapTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--save-every", type=float, default=2.0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--detect-hz", type=float, default=8.0)
    ap.add_argument("--model", default="runs/detect/zhongkui_detector_finetune/weights/best.pt")
    args = ap.parse_args()

    cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy")
    print("启动 scrcpy 流（首次需推送服务端，约 2-5 秒）...")
    cap.start()

    calib, _ = load_calibration()
    sm = MatchStateMachine(minimap_center_norm=calib.get("minimap_center", [0.086, 0.129]))

    print(f"加载检测模型 {args.model} ...")
    det = YoloDetector(args.model, conf=0.35)

    # 小地图跟踪：先验 = 校准点（归一化 -> 首个实际帧尺寸换算）
    mm_prior = None

    def make_tracker(frame):
        nonlocal mm_prior
        if mm_prior is None:
            h, w = frame.shape[:2]
            mc = calib.get("minimap_center", [0.086, 0.129])
            mm_prior = [int(mc[0] * w), int(mc[1] * h)]
        return MinimapTracker(prior_center=mm_prior)

    tracker = None

    save_dir = ROOT / "data" / "live_states"
    last_save = 0.0
    frame_id = 0
    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    t_end = time.time() + args.seconds
    n_frames = 0
    # 对局"确认制"：状态机判定 in_match 后，还需小地图 tracker 连续确认，
    # 防止菜单里的"类小地图"元素（无蓝点）误触发感知与动作
    confirm_streak = 0
    CONFIRM_FRAMES = 2
    confirmed = False
    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")

    print("M1 感知管线运行中...\n")
    try:
        while time.time() < t_end:
            frame, lag_ms = cap.wait_frame(timeout=2.0)
            if frame is None:
                print("  无帧（流异常）")
                continue
            n_frames += 1
            phase = sm.update(frame)
            if phase != MatchPhase.IN_MATCH:
                confirm_streak = 0
                confirmed = False
                if recorder.active:
                    recorder.close()
                    print(f"[{datetime.now():%H:%M:%S}] 对局结束，会话已归档")

            now = time.time()
            if phase == MatchPhase.IN_MATCH and now - last_detect >= detect_interval:
                last_detect = now
                if tracker is None:
                    tracker = make_tracker(frame)
                mm = tracker.update(frame)
                if mm["found"]:
                    confirm_streak += 1
                    if confirm_streak >= CONFIRM_FRAMES:
                        confirmed = True
                else:
                    confirm_streak = 0
                    confirmed = False
                if not confirmed:
                    print(f"[{datetime.now():%H:%M:%S}] 对局确认中（状态机 in_match，"
                          f"小地图待确认 {confirm_streak}/{CONFIRM_FRAMES}）...")
                    continue
                dets = det.detect(frame)
                infer = det.last_infer_ms
                st = build_state(frame, dets, phase.value, minimap={
                    "found": mm["found"],
                    "center": mm["center"],
                    "radius": mm["radius"],
                    "dots": mm["dots"],
                    "towers": mm["towers"],
                }, frame_id=frame_id)
                frame_id += 1
                objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:6]) or "无"
                mm_txt = (f"小地图 {'OK' if mm['found'] else '--'} "
                          f"蓝{len(mm['dots']['blue'])}/红{len(mm['dots']['red'])} "
                          f"({tracker.last_ms:.0f}ms)") if mm["found"] else \
                         f"小地图 未找到 ({tracker.last_ms:.0f}ms)"
                print(f"[{datetime.now():%H:%M:%S}] {phase.value:<9} 采集延迟 {lag_ms:5.0f}ms "
                      f"检测 {infer:5.0f}ms | {objs} | {mm_txt}")
                if not args.no_save:
                    if not recorder.active:
                        recorder.start(meta={"video_source": "scrcpy",
                                             "match_phase_seen": phase.value})
                    if now - last_save >= args.save_every:
                        last_save = now
                        recorder.on_state(st.to_dict())
                        recorder.on_frame(frame)
                if args.show:
                    import cv2
                    vis = frame.copy()
                    for d in dets:
                        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"{d.cls} {d.conf:.2f}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("m1-pipeline", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        print(f"\n完成: 收到 {n_frames} 帧")
    finally:
        if recorder.active:
            recorder.close()
        cap.stop()
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
