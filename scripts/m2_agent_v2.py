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
# 决策参数（v2.1，用户规则指导版）
# ---------------------------------------------------------------------------
SKILL2_THROTTLE_S = 3.0       # 2 技能（钩子）节流：间隔 > 3s（真实冷却由游戏管，节流防重复下发）
SKILL1_THROTTLE_S = 1.5       # 1 技能节流
SUMMONER_THROTTLE_S = 30.0    # 召唤师技能节流（CD 较长）
SKILL3_ACTIVE_S = 3.5         # 大招生效期：三技能释放后 3.5s 内不释放一技能
SKILL_DEBOUNCE_S = 0.05       # 技能释放后防抖：不重复移动
SKILL2_RANGE_FRAC = 0.28      # 二技能（钩子）范围（屏宽比例；由 measure_hook 实测标定 ≈0.30）
HOOK_FLIGHT_S = 0.5           # 钩子飞行期：释放后 0.5s 内停止移动（等勾中结果，防自己走近误判）
NEAR_FRAC = 0.25              # 一技能"身边"阈值（敌人或敌兵 < 0.25 屏宽）
HOOK_CONFIRM_S = 1.0          # 勾中确认窗：二技能释放后 1s 内判定
HOOK_DIST_SHRINK = 0.72       # 勾中判据：敌人距离缩小到释放时的 72% 以下
MOVE_R = 0.8
MOVE_DURATION_MS = 2000       # 持续拖动（用户要求：轮盘移动不是点按而是一直拖动）
# 发育路方向（小地图归一化坐标）：由开局阵营决定（蓝方右下 / 红方左上镜像）
LANE_DIR_BLUE = (0.72, 0.82)
LANE_DIR_RED = (0.28, 0.18)
# 塔规避：屏幕存在敌方塔且无我方小兵时，移动方向朝塔则偏转远离
TURRET_SAFE_FRAC = 0.55       # 塔在屏幕内时与移动目标方向冲突判定阈值


def _debounced(action: dict, now: float, cooldowns: dict) -> dict:
    """技能释放后 50ms 内不再重复移动（防抖）；技能本身不受防抖限制。"""
    if now - float(cooldowns.get("skill", 0.0)) < SKILL_DEBOUNCE_S:
        return {"type": "none", "reason": "skill_debounce"}
    return action


def decide(state_dict: dict, cooldowns: dict) -> dict:
    """规则决策 v2.1（用户策略指导版）：state_dict -> action_dict。

    优先级（用户规则）：
      0. 勾中连招：二技能释放后窗口内敌人被拉近 → 召唤师技能 + 三技能
      1. 塔规避：敌方塔可见且无我方小兵 → 移动避开塔攻击范围（技能不受限）
      2. 二技能：敌方英雄在二技能范围内 → 钩子
      3. 一技能：不在大招生效期 且 敌人/敌兵在身边 → 一技能
      4. 移动：有敌人 → 朝最近敌人持续拖动；无敌人 → 帮发育路射手
         （跟随最近 ally_hero，否则朝发育路方向）

    记忆（cooldowns 扩展字段）：
      skill2_t / skill1_t / skill3_t / summoner_t  最近释放时间
      hook_anchor_dist  二技能释放时最近敌人距离（勾中判定基准）
      hook_pending      二技能释放时刻（勾中确认窗起点）
    """
    now = float(state_dict.get("t") or 0.0)
    w, h = state_dict.get("screen_size") or (1280.0, 720.0)
    aspect = (float(h) / float(w)) if w else 0.5625

    units = state_dict.get("units") or []
    enemies, minions, turrets, allies, ally_minions = [], [], [], [], []
    for u in units:
        cls = str(u.get("cls", ""))
        scr = u.get("screen") or [0.5, 0.5, 0.0, 0.0]
        if cls == "enemy_hero":
            enemies.append(scr)
        elif cls == "enemy_minion":
            minions.append(scr)
        elif cls == "enemy_turret":
            turrets.append(scr)
        elif cls == "ally_hero":
            allies.append(scr)
        elif cls == "ally_minion":
            ally_minions.append(scr)

    def dist_width(cx, cy):
        return math.hypot(cx - 0.5, (cy - 0.5) * aspect)

    def nearest(lst):
        if not lst:
            return None
        return min(lst, key=lambda s: dist_width(s[0], s[1]))

    def ready(key, thr):
        return now - float(cooldowns.get(key, 0.0)) > thr

    # ---- 0) 勾中连招：二技能释放后窗口内，敌人被"拉近" = 勾到了 ----
    hook_pending = float(cooldowns.get("hook_pending", 0.0))
    if hook_pending > 0 and now - hook_pending <= HOOK_CONFIRM_S:
        anchor = float(cooldowns.get("hook_anchor_dist", 0.0))
        ne = nearest(enemies)
        if ne is not None and anchor > 0:
            d = dist_width(ne[0], ne[1])
            # 勾中判据：距离显著缩小（相对+绝对），钩子飞行期 Agent 静止，缩小只能来自钩子
            if d < anchor * HOOK_DIST_SHRINK and (anchor - d) > 0.12:
                cooldowns["hook_pending"] = 0.0
                combo = []
                if ready("summoner_t", SUMMONER_THROTTLE_S):
                    combo.append({"type": "summoner"})
                if ready("skill3_t", SKILL3_ACTIVE_S):
                    combo.append({"type": "skill", "id": 3, "mode": "tap"})
                if combo:
                    cooldowns["summoner_t"] = now
                    cooldowns["skill3_t"] = now
                    cooldowns["skill"] = now
                    return {"type": "combo", "actions": combo, "reason": "hook_confirmed"}
        if now - hook_pending > HOOK_CONFIRM_S:
            cooldowns["hook_pending"] = 0.0

    # ---- 1) 塔规避：敌塔可见且无我方小兵(ally_minion) → 不进入塔攻击范围 ----
    turret_threat = bool(turrets) and not ally_minions
    if turret_threat:
        ne = nearest(enemies)
        nt = nearest(turrets)
        if ne is None and nt is not None:
            # 无敌人时也不朝塔方向走（朝远离塔方向）
            tx, ty = nt[0], nt[1]
            away_theta = math.atan2(-(ty - 0.5) * aspect, -(tx - 0.5))
            return {"type": "move", "theta": away_theta, "r": MOVE_R,
                    "duration_ms": MOVE_DURATION_MS, "reason": "avoid_turret"}
        # 有敌人时：若敌人与塔同向且塔威胁，仍可钩（钩子不进场），移动方向保持但标记
        cooldowns["turret_threat"] = 1.0
    else:
        cooldowns["turret_threat"] = 0.0

    # ---- 2) 二技能：敌方英雄在钩子范围内 ----
    if enemies:
        ne = nearest(enemies)
        d = dist_width(ne[0], ne[1])
        if d <= SKILL2_RANGE_FRAC and ready("skill2_t", SKILL2_THROTTLE_S):
            cooldowns["skill2_t"] = now
            cooldowns["skill"] = now
            cooldowns["hook_pending"] = now
            cooldowns["hook_anchor_dist"] = d
            return {"type": "skill", "id": 2, "mode": "tap", "reason": "enemy_in_skill2_range"}

    # ---- 3) 一技能：不在大招生效期 且 敌人/敌兵在身边 ----
    in_ult = now - float(cooldowns.get("skill3_t", 0.0)) <= SKILL3_ACTIVE_S
    if not in_ult and ready("skill1_t", SKILL1_THROTTLE_S):
        ne = nearest(enemies)
        ne_m = nearest(minions)
        d_e = dist_width(ne[0], ne[1]) if ne else float("inf")
        d_m = dist_width(ne_m[0], ne_m[1]) if ne_m else float("inf")
        if min(d_e, d_m) < NEAR_FRAC:
            cooldowns["skill1_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 1, "mode": "tap",
                    "reason": "enemy_or_minion_near"}

    # ---- 4) 移动：持续拖动 ----
    target = None
    reason = None
    if enemies:
        ne = nearest(enemies)
        target = (ne[0], ne[1])
        reason = "chase_enemy"
    else:
        # 帮发育路射手：跟随最近 ally_hero，否则朝发育路方向（按阵营镜像）
        na = nearest(allies)
        if na is not None:
            target = (na[0], na[1])
            reason = "follow_ally"
        else:
            camp = cooldowns.get("camp") or "blue"
            lane = LANE_DIR_BLUE if camp == "blue" else LANE_DIR_RED
            mm = state_dict.get("minimap") or {}
            if mm.get("found"):
                lx, ly = lane
                target = (lx, ly)
                reason = "lane_develop"
    if target is None:
        return {"type": "none", "reason": "no_target"}
    # 钩子飞行期停止移动（等勾中结果，防自己走近误判连招）
    if now - float(cooldowns.get("skill2_t", 0.0)) < HOOK_FLIGHT_S:
        return {"type": "none", "reason": "hook_flight"}
    theta = math.atan2(-(target[1] - 0.5) * aspect, target[0] - 0.5)
    # 塔规避：若移动方向朝向威胁塔，偏转远离（不进塔范围）
    if turret_threat and cooldowns.get("turret_threat"):
        nt = nearest(turrets)
        if nt is not None:
            tx, ty = nt[0], nt[1]
            t_theta = math.atan2(-(ty - 0.5) * aspect, tx - 0.5)
            diff = abs(((theta - t_theta + math.pi) % (2 * math.pi)) - math.pi)
            if diff < math.pi / 4:  # 目标方向与塔同向
                away = math.atan2(-(ty - 0.5) * aspect, -(tx - 0.5))
                return {"type": "move", "theta": away, "r": MOVE_R,
                        "duration_ms": MOVE_DURATION_MS, "reason": "avoid_turret"}
    move = {"type": "move", "theta": theta, "r": MOVE_R,
            "duration_ms": MOVE_DURATION_MS, "reason": reason}
    return _debounced(move, now, cooldowns)


# ---------------------------------------------------------------------------
# 执行辅助（仅主循环使用，保持 decide 纯函数）
# ---------------------------------------------------------------------------

def update_cooldowns(action: dict, cooldowns: dict, now: float):
    """按已下发的动作推进冷却状态（模拟或真实执行后都调用）。"""
    t = action.get("type")
    if t == "skill":
        sid = int(action.get("id", 0))
        cooldowns[f"skill{sid}_t"] = now
        cooldowns["skill"] = now
    elif t == "summoner":
        cooldowns["summoner_t"] = now
        cooldowns["skill"] = now
    elif t == "combo":
        for sub in action.get("actions", []):
            update_cooldowns(sub, cooldowns, now)


def _rec_hook_measure(cooldowns: dict, state_dict: dict, action: dict, hit: bool):
    """记录钩子释放距离与命中结果（自动标定二技能范围用，写入 data/measure_hook.jsonl）。"""
    try:
        import json
        from pathlib import Path
        now = float(state_dict.get("t") or time.time())
        dist = float(cooldowns.get("hook_anchor_dist", 0.0))
        rec = {"t": now, "dist_frac": round(dist, 4), "hit": hit,
               "reason": action.get("reason", "")}
        p = ROOT / "data" / "measure_hook.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def apply_action(ex, action: dict, cooldowns: dict):
    """真实执行：move / skill_cast / summoner / combo，并推进冷却。"""
    now = time.time()
    t = action.get("type")
    if t == "move":
        ex.move(float(action.get("theta", 0.0)),
                float(action.get("r", MOVE_R)),
                int(action.get("duration_ms", MOVE_DURATION_MS)))
    elif t == "skill":
        ex.skill_cast(int(action.get("id", 0)), action.get("mode", "tap"))
    elif t == "summoner":
        ex.summoner()
    elif t == "combo":
        for i, sub in enumerate(action.get("actions", [])):
            apply_action(ex, sub, cooldowns)
            if i < len(action["actions"]) - 1:
                time.sleep(0.12)  # 连招间隔（召唤师→大招）
    update_cooldowns(action, cooldowns, now)


def format_decision(action: dict) -> str:
    t = action.get("type")
    reason = action.get("reason", "")
    if t == "none":
        return f"无动作 ({reason})"
    if t == "skill":
        return f"技能{action.get('id')} ({action.get('mode', 'tap')}) [{reason}]"
    if t == "summoner":
        return f"召唤师技能 [{reason}]"
    if t == "combo":
        names = [f"{a.get('type')}{a.get('id', '')}" for a in action.get("actions", [])]
        return f"连招 {'→'.join(names)} [{reason}]"
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
    cooldowns = {"skill1_t": 0.0, "skill2_t": 0.0, "skill3_t": 0.0,
                 "summoner_t": 0.0, "skill": 0.0, "hook_pending": 0.0,
                 "hook_anchor_dist": 0.0, "turret_threat": 0.0}

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

            # ---- 开局阵营判断（对局确认后第一帧，泉水颜色）----
            if not cooldowns.get("camp"):
                from wzry.vision.camp import detect_camp_from_center
                camp = detect_camp_from_center(frame)
                if camp:
                    cooldowns["camp"] = camp
                    print(f"[{datetime.now():%H:%M:%S}] 阵营判断: {'蓝方' if camp == 'blue' else '红方'} "
                          f"→ 发育路方向 {LANE_DIR_BLUE if camp == 'blue' else LANE_DIR_RED}")
                else:
                    print(f"[{datetime.now():%H:%M:%S}] 阵营判断: 未判定（泉水色不明显，默认蓝方）")

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

            # ---- 钩子射程测量记录（自动标定二技能范围）----
            if action.get("type") == "skill" and action.get("id") == 2:
                _rec_hook_measure(cooldowns, state_dict, action, hit=False)
            if action.get("type") == "combo":
                _rec_hook_measure(cooldowns, state_dict, action, hit=True)

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
                    if rec.get("type") == "combo":
                        # 连招拆成子动作存档（encode_action 兼容）
                        for sub in rec.get("actions", []):
                            sub_rec = dict(sub)
                            sub_rec["t"] = rec["t"]
                            sub_rec["reason"] = rec.get("reason", "hook_confirmed")
                            recorder.on_action(sub_rec)
                    else:
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
