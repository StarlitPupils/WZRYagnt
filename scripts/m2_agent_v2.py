# -*- coding: utf-8 -*-
"""M2 Agent v2：M1 感知管线 + M2 动作执行器 -> 会操作的规则 Agent（钟馗 v0）。

架构（~10Hz 感知循环）：
    scrcpy 流（30fps）→ MatchStateMachine（对局状态机）
      → 确认制（in_match + 小地图连续 2 帧确认）
      → 感知（YOLO 11 类检测 + MinimapTracker）
      → decide() 规则决策（纯函数，无 IO，便于测试/换策略网络）
      → ActionExecutor 执行（--action 才真正触摸，默认 --no-action 只观察）

规则优先级（v0 简单但安全，见 decide()）：
  1. 钩子可命中（检测到 hook_aim 或 敌方英雄在 2 技能距离内）
     → skill_cast(2, 'tap')，冷却节流 > 3s；
  2. 敌人近（enemy_hero 中心距屏幕中心 < 0.25 屏宽）
     → skill_cast(1, 'tap')，节流 1.5s；
  3. 敌人远 → move(朝向最近敌人, r=0.8, 400ms)；
  4. 无敌人 → 朝兵线方向移动（小地图红点质心方向）；无红点则 idle。
  防抖：任何技能释放后 50ms 内不再下发移动。

用法：
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120              # 只观察（默认，安全开关）
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action     # 真机执行
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action --save   # 执行 + 对局会话采集
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 决策参数（v0）
# ---------------------------------------------------------------------------
SKILL2_THROTTLE_S = 3.0      # 2 技能（钩子）冷却节流：间隔 > 3s
SKILL1_THROTTLE_S = 1.5      # 1 技能节流：间隔 > 1.5s
SKILL_DEBOUNCE_S = 0.05      # 技能释放后 50ms 防抖：不重复移动
SKILL2_RANGE_FRAC = 0.42     # 2 技能可命中距离（相对屏宽，钩子射程较长）
NEAR_FRAC = 0.25             # 敌人"近"阈值：中心距屏幕中心 < 0.25 屏宽
MOVE_R = 0.8                 # 摇杆幅度
MOVE_DURATION_MS = 400       # 移动按住时长（ms）


def _debounced(action: dict, now: float, cooldowns: dict) -> dict:
    """技能释放后 50ms 内不再重复移动（防抖）；技能本身不受防抖限制。"""
    if now - float(cooldowns.get("skill", 0.0)) < SKILL_DEBOUNCE_S:
        return {"type": "none", "reason": "skill_debounce"}
    return action


def decide(state_dict: dict, cooldowns: dict) -> dict:
    """规则决策（纯函数，无 IO）：state_dict -> action_dict。

    state_dict 与 GameState.to_dict() 同构：
      - units: [{"cls": "enemy_hero|hook_aim|...", "screen": [cx, cy, w, h] 归一化}, ...]
      - minimap: {"found": bool, "dots": {"blue": [[nx, ny], ...], "red": [...], ...}}
      - screen_size: [w, h]（用于把纵向距离折算成"屏宽"单位）
      - t: 决策时刻（秒），与 cooldowns 里的时间戳同一时钟
    cooldowns: {"skill1": last_t, "skill2": last_t, "skill": last_t}（秒，0=从未释放）。

    返回 action_dict（与 wzry/train/encoding.py 的 encode_action 兼容）：
      {"type": "skill", "id": 1|2, "mode": "tap", "reason": ...}
      {"type": "move", "theta": 弧度, "r": 0.8, "duration_ms": 400, "reason": ...}
      {"type": "none", "reason": ...}
    """
    now = float(state_dict.get("t") or 0.0)
    w, h = state_dict.get("screen_size") or (1280.0, 720.0)
    aspect = (float(h) / float(w)) if w else 0.5625

    units = state_dict.get("units") or []
    enemies, hook_aim = [], False
    for u in units:
        cls = str(u.get("cls", ""))
        if cls == "enemy_hero":
            enemies.append(u.get("screen") or [0.5, 0.5, 0.0, 0.0])
        elif cls == "hook_aim":
            hook_aim = True

    def dist_width(cx, cy):
        """归一化屏幕坐标 -> 相对屏宽的距离（纵向按 h/w 折算）。"""
        return math.hypot(cx - 0.5, (cy - 0.5) * aspect)

    def ready(key, thr):
        return now - float(cooldowns.get(key, 0.0)) > thr

    if enemies:
        nearest = min(enemies, key=lambda s: dist_width(s[0], s[1]))
        d = dist_width(nearest[0], nearest[1])
        in_s2_range = any(dist_width(s[0], s[1]) <= SKILL2_RANGE_FRAC
                          for s in enemies)
        # 1) 钩子可命中：hook_aim 指示线 或 敌方英雄进入 2 技能距离
        if (hook_aim or in_s2_range) and ready("skill2", SKILL2_THROTTLE_S):
            return {"type": "skill", "id": 2, "mode": "tap",
                    "reason": "hook_aim" if hook_aim else "enemy_in_skill2_range"}
        # 2) 敌人近：中心距屏幕中心 < 0.25 屏宽
        if d < NEAR_FRAC and ready("skill1", SKILL1_THROTTLE_S):
            return {"type": "skill", "id": 1, "mode": "tap", "reason": "enemy_near"}
        # 3) 敌人远：朝最近敌人移动（theta 0=右，逆时针为正，与执行器一致）
        move = {"type": "move",
                "theta": math.atan2(-(nearest[1] - 0.5) * aspect, nearest[0] - 0.5),
                "r": MOVE_R, "duration_ms": MOVE_DURATION_MS,
                "reason": "chase_enemy"}
        return _debounced(move, now, cooldowns)

    # 4) 无敌人：朝兵线方向移动（小地图红点质心方向）；无红点则 idle
    mm = state_dict.get("minimap") or {}
    red = []
    if mm.get("found"):
        red = (mm.get("dots") or {}).get("red") or []
    if red:
        cx = sum(p[0] for p in red) / len(red)
        cy = sum(p[1] for p in red) / len(red)
        move = {"type": "move",
                "theta": math.atan2(-(cy - 0.5), cx - 0.5),
                "r": MOVE_R, "duration_ms": MOVE_DURATION_MS,
                "reason": "lane_red_centroid"}
        return _debounced(move, now, cooldowns)
    return {"type": "none", "reason": "no_target"}


# ---------------------------------------------------------------------------
# 执行辅助（仅主循环使用，保持 decide 纯函数）
# ---------------------------------------------------------------------------

def update_cooldowns(action: dict, cooldowns: dict, now: float):
    """按已下发的动作推进冷却状态（模拟或真实执行后都调用）。"""
    if action.get("type") == "skill":
        sid = int(action.get("id", 0))
        if sid in (1, 2):
            cooldowns[f"skill{sid}"] = now
        cooldowns["skill"] = now


def apply_action(ex, action: dict, cooldowns: dict):
    """真实执行：move / skill_cast，并推进冷却。"""
    now = time.time()
    t = action.get("type")
    if t == "move":
        ex.move(float(action.get("theta", 0.0)),
                float(action.get("r", MOVE_R)),
                int(action.get("duration_ms", MOVE_DURATION_MS)))
    elif t == "skill":
        ex.skill_cast(int(action.get("id", 0)), action.get("mode", "tap"))
    update_cooldowns(action, cooldowns, now)


def format_decision(action: dict) -> str:
    t = action.get("type")
    reason = action.get("reason", "")
    if t == "none":
        return f"无动作 ({reason})"
    if t == "skill":
        return f"技能{action.get('id')} ({action.get('mode', 'tap')}) [{reason}]"
    if t == "move":
        return (f"移动 θ={action.get('theta', 0.0):+.2f} r={action.get('r', 0.0)} "
                f"{action.get('duration_ms', 0)}ms [{reason}]")
    return f"{t} [{reason}]"


def main():
    ap = argparse.ArgumentParser(description="M2 Agent v2：规则 Agent（感知→决策→执行）")
    ap.add_argument("--seconds", type=float, default=120.0, help="运行时长（秒），默认 120")
    ap.add_argument("--model", default=str(ROOT / "runs" / "detect" / "zhongkui_11cls"
                                           / "weights" / "best.pt"),
                    help="YOLO 11 类模型路径")
    ap.add_argument("--action", action="store_true",
                    help="真正执行动作（默认只观察不触摸，防误操作）")
    ap.add_argument("--no-action", dest="no_action", action="store_true",
                    help="只观察（默认，安全开关）；与 --action 同时给出时 no-action 优先")
    ap.add_argument("--save", action="store_true", help="对局会话采集（states/actions.jsonl）")
    ap.add_argument("--detect-hz", type=float, default=10.0, help="感知循环频率，默认 10Hz")
    ap.add_argument("--show", action="store_true", help="OpenCV 窗口实时预览")
    args = ap.parse_args()

    do_action = bool(args.action) and not args.no_action

    from wzry.calib import load_calibration
    from wzry.capture.scrcpy_stream import ScrcpyStreamCapture
    from wzry.data.collector import MatchRecorder
    from wzry.state.fuser import build_state
    from wzry.state.match_state import MatchPhase, MatchStateMachine
    from wzry.vision.detector import YoloDetector
    from wzry.vision.minimap_tracker import MinimapTracker

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

    if do_action:
        from wzry.action.executor_v2 import ActionExecutor
        ex = ActionExecutor()
        print("动作模式: 执行（--action，真机触摸）")
    else:
        ex = None
        print("动作模式: 仅观察（默认 --no-action，不注入任何触摸；加 --action 才执行）")

    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")
    cooldowns = {"skill1": 0.0, "skill2": 0.0, "skill": 0.0}

    # 确认制：状态机判定 in_match 后，还需小地图 tracker 连续 2 帧确认
    CONFIRM_FRAMES = 2
    confirm_streak = 0
    confirmed = False

    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    last_log = 0.0
    last_sig = ("none", None)
    frame_id = 0
    n_frames = 0
    n_ticks = 0
    n_actions = 0
    infer_sum = 0.0
    t_end = time.time() + args.seconds

    print("M2 Agent v2 运行中（Ctrl+C 退出）...\n")
    try:
        while time.time() < t_end:
            frame, lag_ms = cap.wait_frame(timeout=2.0)
            if frame is None:
                print("  无帧（流异常）")
                continue
            n_frames += 1
            phase = sm.update(frame)
            now = time.time()
            if phase != MatchPhase.IN_MATCH:
                confirm_streak = 0
                confirmed = False
                if recorder.active:
                    recorder.close()
                    print(f"[{datetime.now():%H:%M:%S}] 对局结束，会话已归档")
                continue
            if now - last_detect < detect_interval:
                continue
            last_detect = now

            # ---- 确认制 ----
            if tracker is None:
                tracker = make_tracker(frame)
            mm = tracker.update(frame)
            if not mm["found"]:
                confirm_streak = 0
                confirmed = False
                print(f"[{datetime.now():%H:%M:%S}] 对局确认中（小地图暂未定位）...")
                continue
            confirm_streak += 1
            if confirm_streak < CONFIRM_FRAMES:
                confirmed = False
                print(f"[{datetime.now():%H:%M:%S}] 对局确认中 {confirm_streak}/{CONFIRM_FRAMES} ...")
                continue
            confirmed = True

            # ---- 感知 ----
            dets = det.detect(frame)
            st = build_state(frame, dets, phase.value, minimap={
                "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
                "dots": mm["dots"], "towers": mm["towers"],
            }, frame_id=frame_id)
            frame_id += 1
            st.t = time.time()  # 决策时刻
            state_dict = st.to_dict()
            n_ticks += 1
            infer_sum += det.last_infer_ms

            # ---- 规则决策 ----
            action = decide(state_dict, cooldowns)
            sig = (action.get("type"), action.get("id"))
            if sig != last_sig:
                print(f"[{datetime.now():%H:%M:%S}] [决策] {format_decision(action)}")
                last_sig = sig

            # ---- 执行 / 模拟 ----
            if action.get("type") != "none":
                if do_action:
                    apply_action(ex, action, cooldowns)
                    n_actions += 1
                else:
                    update_cooldowns(action, cooldowns, now)  # 观察模式也推进冷却，便于观察节流

            # ---- 采集 ----
            if args.save:
                if not recorder.active:
                    recorder.start(meta={"agent": "m2_agent_v2", "model": str(args.model),
                                         "action": "on" if do_action else "off"})
                recorder.on_state(state_dict)
                if do_action and action.get("type") != "none":
                    rec = dict(action)
                    rec["t"] = time.time()
                    recorder.on_action(rec)

            # ---- 周期日志 / 预览 ----
            if now - last_log >= 0.5:
                last_log = now
                objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:5]) or "无"
                mm_txt = (f"小地图 蓝{len(mm['dots']['blue'])}/红{len(mm['dots']['red'])} "
                          f"({tracker.last_ms:.0f}ms)") if mm["found"] else "小地图 未找到"
                print(f"[{datetime.now():%H:%M:%S}] 检测 {det.last_infer_ms:5.0f}ms | "
                      f"{objs} | {mm_txt} | 决策: {format_decision(action)}")
            if args.show:
                import cv2
                vis = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{d.cls} {d.conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if mm["found"]:
                    from wzry.vision.minimap import draw_overlay
                    vis = draw_overlay(vis, mm)
                cv2.putText(vis, f"act: {format_decision(action)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("m2-agent-v2", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(f"\n完成: 帧 {n_frames}，感知 {n_ticks} 次，"
              f"平均检测 {infer_sum / max(1, n_ticks):.0f}ms，"
              f"执行动作 {n_actions} 次（模式: {'执行' if do_action else '仅观察'}）")
    except KeyboardInterrupt:
        print("\n手动退出。")
    finally:
        if recorder.active:
            recorder.close()
        cap.stop()
        if args.show:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
