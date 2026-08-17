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
from wzry.state.fuser import build_state  # noqa: E402
from wzry.state.match_state import MatchPhase, MatchStateMachine  # noqa: E402
from wzry.vision.detector import YoloDetector  # noqa: E402


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

    save_dir = ROOT / "data" / "live_states"
    last_save = 0.0
    frame_id = 0
    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    t_end = time.time() + args.seconds
    n_frames = 0

    print("M1 感知管线运行中...\n")
    try:
        while time.time() < t_end:
            frame, lag_ms = cap.wait_frame(timeout=2.0)
            if frame is None:
                print("  无帧（流异常）")
                continue
            n_frames += 1
            phase = sm.update(frame)

            now = time.time()
            if now - last_detect >= detect_interval:
                last_detect = now
                dets = det.detect(frame)
                infer = det.last_infer_ms
                st = build_state(frame, dets, phase.value, frame_id=frame_id)
                frame_id += 1
                objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:6]) or "无"
                print(f"[{datetime.now():%H:%M:%S}] {phase.value:<9} 采集延迟 {lag_ms:5.0f}ms "
                      f"检测 {infer:5.0f}ms | {objs}")
                if phase == MatchPhase.IN_MATCH and not args.no_save and now - last_save >= args.save_every:
                    last_save = now
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fn = save_dir / f"state_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
                    fn.write_text(st.to_json(), encoding="utf-8")
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
        cap.stop()
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
