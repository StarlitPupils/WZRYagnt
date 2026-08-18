# -*- coding: utf-8 -*-
"""示范局录制器：用户手动操作一局时，并行记录三路数据，供事后复盘讲解决策。

用户流程：
    1) 进对局前启动本脚本（--seconds 给足时长，或 Ctrl+C 提前结束）
    2) 用户正常手动打完一局（脚本完全不注入任何触摸，纯旁观）
    3) 结束后运行 scripts/train/infer_actions.py 反推用户动作，
       再用 analyze_session / 对齐工具复盘，用户逐段讲解决策。

记录内容（data/demos/<session>/）：
    video.mkv          整局录屏（scrcpy --record，1280x720）
    states.jsonl       GameState 流（检测 11 类 + 小地图 + UI，5Hz）
    touch.jsonl        getevent 触摸事件原始流（物理坐标，尽力而为；不可用则文件为空）
    manifest.json      会话元信息（含 wall-clock 对齐基准 t0 = 录像起点时刻）
    frames/            关键帧 PNG（默认每 3 秒一帧，便于快速浏览）

用法：
    venv\\Scripts\\python.exe scripts\\demo_record.py --seconds 900
    venv\\Scripts\\python.exe scripts\\demo_record.py --seconds 900 --no-touch
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# 管道/重定向下输出 UTF-8（避免 GBK 乱码）并即时刷新
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_ffmpeg():
    for cand in (ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
                 ROOT / "tools" / "ffmpeg.exe",
                 Path(r"E:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe")):
        if cand.exists():
            return cand
    return None


def discover_mumu_adb():
    from wzry.control.executor import discover_mumu_adb_devices  # noqa: E402
    serials = discover_mumu_adb_devices()
    return serials[0] if serials else None


class TouchLogger(threading.Thread):
    """后台线程：adb shell getevent 捕获触摸/摇杆设备事件，逐行写入 touch.jsonl。

    事件格式（一行一个）：
        {"t": wall_clock, "dev": "/dev/input/event4", "kind": "touch",
         "raw": "0003 0035 000002b2"}
    kind: touch=鼠标/触摸屏（物理坐标，横屏需旋转映射）；joystick=WASD 摇杆
    """

    def __init__(self, serial, out_path, adb="adb"):
        super().__init__(daemon=True)
        self.serial = serial
        self.out_path = Path(out_path)
        self.adb = adb
        self._stop = False
        self._fp = None
        self._lock = threading.Lock()

    def _find_input_devs(self):
        """在 getevent -pl 输出中找触摸设备（ABS_MT）与摇杆设备（WASD/Joystick）。"""
        try:
            out = subprocess.run(
                [self.adb, "-s", self.serial, "shell", "getevent", "-pl"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30)
            txt = out.stdout + out.stderr
        except Exception:
            return []
        devs = []
        blocks = txt.split("add device")
        for b in blocks[1:]:
            lines = b.splitlines()
            if not lines:
                continue
            dev = "/dev/input/" + lines[0].split("/")[-1].strip()
            if "ABS_MT_POSITION_X" in b and "ABS_MT_POSITION_Y" in b:
                devs.append((dev, "touch"))
            elif "BTN_GAMEPAD" in b or ("Joystick" in b and "ABS_X" in b):
                devs.append((dev, "joystick"))
        return devs

    def run(self):
        try:
            devs = self._find_input_devs()
            if not devs:
                print(f"[touch] 未找到输入设备（getevent -pl 无 ABS_MT/Joystick），触摸流停用")
                return
            print(f"[touch] 捕获设备: {', '.join(f'{d}({k})' for d, k in devs)}")
            self._fp = open(self.out_path, "a", encoding="utf-8")

            def reader(dev, kind, proc):
                try:
                    for line in proc.stdout:
                        if self._stop:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        rec = {"t": time.time(), "dev": dev, "kind": kind,
                               "raw": line.split("]", 1)[-1].strip()}
                        with self._lock:
                            self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            threads = []
            for dev, kind in devs:
                try:
                    proc = subprocess.Popen(
                        [self.adb, "-s", self.serial, "shell", "getevent", "-t", dev],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace")
                except Exception:
                    continue
                t = threading.Thread(target=reader, args=(dev, kind, proc), daemon=True)
                t.start()
                threads.append(t)
            while not self._stop:
                time.sleep(0.5)
            for t in threads:
                t.join(timeout=1.0)
        except Exception as e:
            print(f"[touch] 触摸流异常: {e}")
        finally:
            if self._fp:
                self._fp.close()

    def stop(self):
        self._stop = True


def main():
    ap = argparse.ArgumentParser(description="示范局录制器（旁观用户手动操作）")
    ap.add_argument("--seconds", type=float, default=900.0, help="最大录制时长")
    ap.add_argument("--end-idle", type=float, default=20.0,
                    help="对局结束（非 in_match）持续该秒数后自动停止录制")
    ap.add_argument("--model", default=str(ROOT / "runs" / "detect" / "zhongkui_11cls"
                                           / "weights" / "best.pt"))
    ap.add_argument("--detect-hz", type=float, default=5.0, help="感知频率")
    ap.add_argument("--frame-every", type=float, default=3.0, help="关键帧间隔秒")
    ap.add_argument("--no-touch", action="store_true", help="不捕获触摸事件")
    ap.add_argument("--serial", default=None, help="adb serial（默认自动发现 MuMu）")
    args = ap.parse_args()

    from wzry.capture.scrcpy_stream import ScrcpyStreamCapture  # noqa: E402
    from wzry.data.collector import MatchRecorder  # noqa: E402
    from wzry.state.fuser import build_state  # noqa: E402
    from wzry.state.match_state import MatchPhase, MatchStateMachine  # noqa: E402
    from wzry.vision.detector import YoloDetector  # noqa: E402
    from wzry.vision.minimap_tracker import MinimapTracker  # noqa: E402
    from wzry.vision.ui_reader import find_hp_bar, read_ui  # noqa: E402

    serial = args.serial or discover_mumu_adb()
    if not serial:
        print("错误：未发现 MuMu ADB 设备（请确认模拟器已启动）")
        return 1
    print(f"[adb] 设备 {serial}")

    # ---- 会话目录 ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = ROOT / "data" / "demos" / ts
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "frames").mkdir(exist_ok=True)

    # ---- scrcpy 整局录像（单文件，不轮转）----
    cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy", serial=serial,
                              temp_dir=sess, rotate_s=10 ** 6)
    print("启动 scrcpy 录像（首次推送服务端约 2-5 秒）...")
    cap.start()
    video_path = None
    # ScrcpyStreamCapture 内部文件在 temp_dir/stream_0001.mkv
    for f in sorted(sess.glob("stream_*.mkv")):
        video_path = f
    t0 = cap._session_start  # 录像起点 wall clock（对齐基准）
    print(f"[video] 录像文件 {video_path}（t0={t0:.3f}）")

    # ---- 触摸流 ----
    if not args.no_touch:
        touch = TouchLogger(serial, sess / "touch.jsonl")
        touch.start()

    # ---- 感知管线 ----
    from wzry.calib import load_calibration
    calib, _ = load_calibration()
    sm = MatchStateMachine(minimap_center_norm=calib.get("minimap_center", [0.086, 0.129]))
    det = YoloDetector(args.model, conf=0.35)
    mc = calib.get("minimap_center", [0.086, 0.129])
    tracker = None
    skills = None
    with open(ROOT / "configs" / "calibration_absolute.json", encoding="utf-8") as f:
        pts = json.load(f)["points"]
    skills = {1: pts["skill1"], 2: pts["skill2"], 3: pts["skill3"]}

    recorder = MatchRecorder(base_dir=sess.parent, frame_every_s=args.frame_every)
    recorder.start(meta={"source": "demo", "t0": t0, "serial": serial,
                         "model": args.model, "action": "human"},
                   session_name=sess.name)

    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    last_hp = 0.0
    frame_id = 0
    n_ticks = 0
    in_match = False
    idle_since = None
    t_end = time.time() + args.seconds
    print("示范录制中：请进入对局手动操作（WASD 移动 + 鼠标其余，脚本只旁观不注入）。"
          "Ctrl+C 结束；对局结束约 20 秒后自动停止。")

    try:
        while time.time() < t_end:
            frame, _ = cap.wait_frame(timeout=2.0)
            if frame is None:
                continue
            now = time.time()
            phase = sm.update(frame)
            if phase == MatchPhase.IN_MATCH:
                if not in_match:
                    print(f"[{datetime.now():%H:%M:%S}] 对局开始（状态机确认）")
                    in_match = True
                idle_since = None
            elif in_match:
                print(f"[{datetime.now():%H:%M:%S}] 对局结束（回到非对局画面）")
                in_match = False
                idle_since = now
            if not in_match:
                if idle_since is not None and now - idle_since > args.end_idle:
                    print("对局结束，自动停止录制。")
                    break
                continue
            if now - last_detect < detect_interval:
                continue
            last_detect = now
            if tracker is None:
                h, w = frame.shape[:2]
                tracker = MinimapTracker(prior_center=[int(mc[0] * w), int(mc[1] * h)])
            mm = tracker.update(frame)
            dets = det.detect(frame)
            st = build_state(frame, dets, phase.value, minimap={
                "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
                "dots": mm["dots"], "towers": mm["towers"],
            }, frame_id=frame_id)
            frame_id += 1
            st.t = now
            state_dict = st.to_dict()
            if now - last_hp >= 0.5:
                hp_res = find_hp_bar(frame)
                last_hp = now
                if hp_res:
                    state_dict.setdefault("ui", {})["hp"] = hp_res[-1]
            ui = read_ui(frame, skills)
            state_dict.setdefault("ui", {}).setdefault("skill_dark", {})
            for k, v in ui["skills"].items():
                if v["dark_frac"] is not None:
                    state_dict["ui"]["skill_dark"][k] = v["dark_frac"]
            if ui["skill_states"]:
                state_dict.setdefault("ui", {})["skill_states"] = ui["skill_states"]
            if ui["mp_bar"]:
                state_dict.setdefault("ui", {})["mp"] = ui["mp_bar"][-1]
            recorder.on_state(state_dict)
            recorder.on_frame(frame)
            n_ticks += 1
            if n_ticks % 25 == 0:
                objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:4]) or "无"
                mm_txt = (f"蓝{len(mm['dots']['blue'])}/红{len(mm['dots']['red'])}"
                          if mm["found"] else "未定位")
                print(f"[{datetime.now():%H:%M:%S}] 帧{frame_id} {objs} | 小地图 {mm_txt}"
                      f" | HP {state_dict.get('ui', {}).get('hp', '?')}")
    except KeyboardInterrupt:
        print("\n手动结束录制。")
    finally:
        if not args.no_touch:
            touch.stop()
        cap.stop()
        recorder.close()

    # ---- 收尾：修复 mkv + 写 manifest ----
    import cv2
    if video_path and video_path.exists() and video_path.stat().st_size > 0:
        fixed = sess / "video.mkv"
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            print("[video] 未找到 ffmpeg，跳过修复")
        else:
            try:
                subprocess.run(
                    [str(ffmpeg), "-y", "-i", str(video_path), "-c", "copy",
                     "-movflags", "+faststart", str(fixed)],
                    capture_output=True, timeout=120)
                if fixed.exists() and fixed.stat().st_size > 1000:
                    vcap = cv2.VideoCapture(str(fixed))
                    fps = vcap.get(cv2.CAP_PROP_FPS)
                    n_f = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                    vcap.release()
                    video_path = fixed
                    print(f"[video] 已修复并整理为 {fixed}（fps={fps:.2f} 帧数={n_f}）")
            except Exception as e:
                print(f"[video] mkv 修复失败（{e}），保留原始 {video_path}")
    manifest = {
        "session": ts,
        "started_at": datetime.fromtimestamp(t0).isoformat(),
        "t0": t0,
        "video": str(video_path) if video_path else None,
        "n_states": n_ticks,
        "duration_s": round(time.time() - t0, 2),
        "serial": serial,
        "note": "t0 为录像起点 wall clock；录像秒 s ↔ wall t0+s。",
    }
    (sess / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 示范会话完成: {sess} ===")
    print(f"  状态流 {n_ticks} 条 | 录像 {video_path}")
    print("下一步：")
    print(f"  venv\\Scripts\\python.exe scripts\\train\\infer_actions.py "
          f"--video {video_path} --out {sess / 'actions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
