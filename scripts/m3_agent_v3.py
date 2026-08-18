# -*- coding: utf-8 -*-
"""M3 Agent v3：感知管线 + BCNet 策略（替代 v2 规则决策）-> 会自己玩的钟馗。

与 m2_agent_v2 同架构，决策核心换成 wzry.policy.inference.BCNetInference：
    scrcpy 流 → 状态机+确认制 → 感知(11类检测+小地图) → GameState
      → BCNet 推理（move 8向/动作类/幅度）→ 动作解码 → ActionExecutor 执行
      → MatchRecorder 会话采集（states+actions）

保留安全机制：
  - 默认 --no-action 只观察（打印模型决策）
  - 技能冷却节流（2技能 3s / 1技能 1.5s）与防抖（技能后 50ms 不移动）
  - 模型输出 none 时不动作

用法：
    venv\\Scripts\\python.exe scripts\\m3_agent_v3.py --checkpoint runs\\bc\\bc_agent_v0.pt --seconds 120
    venv\\Scripts\\python.exe scripts\\m3_agent_v3.py --checkpoint runs\\bc\\bc_agent_v0.pt --seconds 120 --action --save
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKILL2_THROTTLE_S = 3.0
SKILL1_THROTTLE_S = 1.5
SKILL_DEBOUNCE_S = 0.05
MOVE_DURATION_MS = 400


def apply_guards(action: dict, now: float, cooldowns: dict) -> dict:
    """冷却节流 + 技能后防抖 + 回城/恢复节流（模型动作的外部安全层）。"""
    if action.get("type") == "skill":
        key = f"skill{action.get('id')}"
        thr = SKILL2_THROTTLE_S if action.get("id") == 2 else SKILL1_THROTTLE_S
        if now - float(cooldowns.get(key, 0.0)) <= thr:
            return {"type": "none", "reason": f"{key}_throttle"}
        cooldowns[key] = now
        cooldowns["skill"] = now
        return action
    if action.get("type") == "recall":
        if now - float(cooldowns.get("recall_t", 0.0)) <= 20.0:
            return {"type": "none", "reason": "recall_throttle"}
        cooldowns["recall_t"] = now
        return action
    if action.get("type") == "restore":
        if now - float(cooldowns.get("restore_t", 0.0)) <= 10.0:
            return {"type": "none", "reason": "restore_throttle"}
        cooldowns["restore_t"] = now
        return action
    if action.get("type") == "summoner":
        if now - float(cooldowns.get("summoner_t", 0.0)) <= 30.0:
            return {"type": "none", "reason": "summoner_throttle"}
        cooldowns["summoner_t"] = now
        return action
    if action.get("type") == "move":
        if now - float(cooldowns.get("skill", 0.0)) < SKILL_DEBOUNCE_S:
            return {"type": "none", "reason": "skill_debounce"}
        action.setdefault("duration_ms", MOVE_DURATION_MS)
    return action


def build_state(frame, dets, mm, phase, frame_id, ui=None):
    """感知输出 -> GameState dict（供 BCNet 编码与会话采集）。"""
    from wzry.state.fuser import build_state
    st = build_state(frame, dets, phase, minimap={
        "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
        "dots": mm["dots"], "towers": mm["towers"]}, frame_id=frame_id)
    if ui:
        st.ui = ui
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="BCNet 权重 (.pt)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--model", default="runs/detect/zhongkui_11cls/weights/best.pt")
    ap.add_argument("--action", action="store_true", help="真机执行（默认只观察）")
    ap.add_argument("--no-action", dest="action", action="store_false")
    ap.set_defaults(action=False)
    ap.add_argument("--save", action="store_true", help="对局会话采集")
    ap.add_argument("--detect-hz", type=float, default=8.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    from wzry.calib import load_calibration
    from wzry.capture.scrcpy_stream import ScrcpyStreamCapture
    from wzry.data.collector import MatchRecorder
    from wzry.policy.inference import BCNetInference
    from wzry.state.match_state import MatchPhase, MatchStateMachine
    from wzry.vision.detector import YoloDetector
    from wzry.vision.minimap_tracker import MinimapTracker

    cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy")
    print("启动 scrcpy 流...")
    cap.start()

    calib, _ = load_calibration()
    sm = MatchStateMachine(minimap_center_norm=calib.get("minimap_center", [0.086, 0.129]))
    print(f"加载 11 类检测模型 {args.model} ...")
    det = YoloDetector(args.model, conf=0.35)
    print(f"加载 BCNet 策略 {args.checkpoint} ...")
    policy = BCNetInference(args.checkpoint)

    tracker = None
    mm_prior = None
    confirm = 0
    confirmed = False
    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")
    cooldowns = {"skill1": 0.0, "skill2": 0.0, "skill": 0.0,
                 "recall_t": 0.0, "restore_t": 0.0, "summoner_t": 0.0}
    frame_id = 0
    n_action = 0
    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    t_end = time.time() + args.seconds

    print(f"动作模式: {'执行' if args.action else '仅观察（--action 才真机触摸）'}")
    print("M3 Agent v3 运行中（Ctrl+C 退出）...\n")

    try:
        while time.time() < t_end:
            frame, _ = cap.wait_frame(timeout=2.0)
            if frame is None:
                print("  无帧（流异常）")
                continue
            phase = sm.update(frame)
            if phase != MatchPhase.IN_MATCH:
                confirm = 0
                confirmed = False
                if recorder.active:
                    recorder.close()
                continue
            now = time.time()
            if now - last_detect < detect_interval:
                continue
            last_detect = now
            if tracker is None:
                h, w = frame.shape[:2]
                mc = calib.get("minimap_center", [0.086, 0.129])
                tracker = MinimapTracker(prior_center=[int(mc[0] * w), int(mc[1] * h)])
            mm = tracker.update(frame)
            if not mm["found"]:
                print(f"[{datetime.now():%H:%M:%S}] 对局确认中（小地图暂未定位）...")
                continue
            confirm += 1
            if not confirmed:
                if confirm < 2:
                    print(f"[{datetime.now():%H:%M:%S}] 对局确认中 {confirm}/2 ...")
                    continue
                confirmed = True
                print(f"[{datetime.now():%H:%M:%S}] 对局确认！BCNet 策略接管...")

            dets = det.detect(frame)
            st = build_state(frame, dets, mm, phase.value, frame_id)
            frame_id += 1

            act = policy.decide(st.to_dict())
            action = {"type": "none"}
            if act.get("skill_id"):
                action = {"type": "skill", "id": act["skill_id"], "mode": "tap",
                          "reason": act.get("reason", "bc_skill")}
            elif act.get("attack"):
                action = {"type": "attack", "priority": "free", "reason": "bc_attack"}
            elif act.get("reason", "").startswith("bc:") and act.get("reason") != "bc:none":
                pass  # none/其他类不动作
            elif act.get("move_bin") is not None:
                action = {"type": "move", "theta": act["theta"], "r": act["r"],
                          "reason": f"bc:move_bin{act['move_bin']}"}
            action = apply_guards(action, now, cooldowns)

            objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:5]) or "无"
            print(f"[{datetime.now():%H:%M:%S}] 检测 {det.last_infer_ms:4.0f}ms "
                  f"策略 {policy.last_infer_ms:4.0f}ms | {objs} | "
                  f"小地图 蓝{len(mm['dots']['blue'])}/红{len(mm['dots']['red'])} "
                  f"| 决策: {action.get('type')} ({action.get('reason','')})")

            if args.action and action["type"] != "none":
                ex = getattr(main, "_ex", None)
                if ex is None:
                    from wzry.action.executor_v2 import ActionExecutor
                    ex = ActionExecutor()
                    main._ex = ex
                if action["type"] == "skill":
                    ex.skill_cast(action["id"], action.get("mode", "tap"))
                elif action["type"] == "move":
                    ex.move(action["theta"], action.get("r", 0.8), action.get("duration_ms", 400))
                elif action["type"] == "attack":
                    ex.attack("free")
                elif action["type"] == "recall":
                    ex.recall()
                elif action["type"] == "restore":
                    ex.restore()
                elif action["type"] == "summoner":
                    ex.summoner()
                n_action += 1
                act_log = {"t": time.time(), **action}
                if args.save:
                    if not recorder.active:
                        recorder.start(meta={"agent": "v3", "checkpoint": str(args.checkpoint)})
                    recorder.on_state(st.to_dict())
                    recorder.on_action(act_log)
            elif args.save and confirmed:
                if not recorder.active:
                    recorder.start(meta={"agent": "v3", "checkpoint": str(args.checkpoint)})
                recorder.on_state(st.to_dict())
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        if recorder.active:
            recorder.close()
        cap.stop()


if __name__ == "__main__":
    main()
