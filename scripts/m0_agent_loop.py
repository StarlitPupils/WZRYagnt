# -*- coding: utf-8 -*-
"""M0 被动值守 Agent（单模拟器，人工进房）。

职责：
  1. 以 ~2Hz 采集设备画面（默认 adb screencap，窗口最小化也可用）；
  2. MatchStateMachine 判断是否已在对局中；
  3. 对局外：安静等待，提示人工进房（默认不注入任何输入）；
  4. 对局中：进入采集模式——定期把 GameState JSON 存入 data/live_states/，
     并打印帧信息（供 M1 感知开发联调）。

用法：
    venv\\Scripts\\python.exe scripts\\m0_agent_loop.py [--hz 2] [--save-every 2.0]
        [--no-save] [--show] [--demo-tap] [--capture adb|window] [--restore-window]

参数：
    --hz             观测频率（默认 2）
    --save-every     对局中状态存档间隔秒（默认 2.0）
    --no-save        不存档
    --show           用 OpenCV 窗口实时预览（需 cv2，已装）
    --demo-tap       找到设备后演示一次安全空指令（keyevent 0），验证控制通道
    --capture        采集方式：adb（默认，设备帧缓冲）| window（MuMu 窗口 mss）
    --restore-window 恢复模拟器窗口（最小化时画面不可见，恢复后可在桌面看到游戏）
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.calib import load_calibration  # noqa: E402
from wzry.control.executor import AdbExecutor, discover_mumu_adb_devices  # noqa: E402
from wzry.state.match_state import MatchPhase, MatchStateMachine  # noqa: E402
from wzry.state.schema import GameState  # noqa: E402


def build_state_machine():
    calib, src = load_calibration()
    mc = calib.get("minimap_center")
    if not mc:
        print(f"警告: 校准文件 {src} 缺少 minimap_center，使用默认 (0.086, 0.129)")
        mc = [0.086, 0.129]
    return MatchStateMachine(minimap_center_norm=mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--save-every", type=float, default=2.0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--demo-tap", action="store_true")
    ap.add_argument("--capture", choices=["adb", "window"], default="adb")
    ap.add_argument("--restore-window", action="store_true")
    args = ap.parse_args()

    if args.capture == "window":
        from wzry.capture.window import WindowCapture
        cap = WindowCapture()
        print("采集方式: MuMu 窗口 mss（窗口需可见）")
    else:
        from wzry.capture.adb import AdbCapture
        cap = AdbCapture()
        print("采集方式: adb screencap（设备帧缓冲，窗口最小化也可用）")

    if args.restore_window:
        try:
            import win32gui
            hw = win32gui.FindWindow(None, "MuMu模拟器12")
            if hw:
                win32gui.ShowWindow(hw, 9)  # SW_RESTORE
                print("已恢复 MuMu 模拟器窗口")
        except Exception as e:
            print(f"恢复窗口失败: {e}")

    sm = build_state_machine()

    print("M0 被动值守 Agent 启动。请手动打开王者荣耀并进入对局，Agent 会自动接管观察。")
    print("（默认不注入任何输入；Ctrl+C 退出）\n")

    if args.demo_tap:
        serials = discover_mumu_adb_devices()
        if serials:
            ex = AdbExecutor(serial=serials[0], event_log=ROOT / "logs" / "m0_events.jsonl")
            t0 = time.perf_counter()
            ok = ex.key(0, source="demo")
            print(f"控制通道验证: keyevent 0 -> {'OK' if ok else 'FAIL'} "
                  f"({(time.perf_counter()-t0)*1000:.0f} ms, 设备 {serials[0]})")
            ex.close()
        else:
            print("控制通道验证: 无 ADB 设备，跳过（启动模拟器后可用 --demo-tap 验证）")

    save_dir = ROOT / "data" / "live_states"
    last_save = 0.0
    last_phase_print = ""
    frame_id = 0
    interval = 1.0 / max(0.5, args.hz)

    try:
        while True:
            frame, cap_ms = cap.get_frame()
            if frame is None:
                if last_phase_print != "no_window":
                    print(f"[{datetime.now():%H:%M:%S}] 未找到 MuMu 窗口，等待模拟器启动...")
                    last_phase_print = "no_window"
                time.sleep(1.0)
                continue

            phase = sm.update(frame)
            h, w = frame.shape[:2]

            if phase != last_phase_print:
                print(f"[{datetime.now():%H:%M:%S}] 阶段切换 -> {phase.value} "
                      f"(画面 {w}x{h}, 采集 {cap_ms:.0f} ms)")
                last_phase_print = phase

            if phase == MatchPhase.IN_MATCH:
                now = time.time()
                if not args.no_save and now - last_save >= args.save_every:
                    st = GameState(t=now, phase=phase.value, frame_id=frame_id,
                                   source="live", screen_size=[w, h])
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fn = save_dir / f"state_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
                    fn.write_text(st.to_json(), encoding="utf-8")
                    last_save = now
                    frame_id += 1
                if args.show:
                    import cv2
                    cv2.imshow("wzry-agent", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        cap.close()
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
