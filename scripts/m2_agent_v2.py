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
  3. 敌人贴身（< 0.20 屏宽）且技能不可用 → 普攻（攻击键），节流 1s；
  4. 敌人远 → move(朝向最近敌人, r=0.8, 400ms)；
  5. 无敌人 → 朝兵线方向移动（小地图红点质心方向）；无红点则 idle。
  防抖：任何技能释放后 50ms 内不再下发移动。
  塔：被塔攻击（塔中心 < 0.40 屏宽且无我方小兵）→ 立即反向逃离；

用法：
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120              # 只观察（默认，安全开关）
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action     # 真机执行
    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action --save   # 执行 + 对局会话采集
"""
import argparse
import json
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
MOVE_DURATION_MS = 2500       # 持续拖动（swipe 阻塞期间主循环暂停，不宜过长）
# 发育路方向（小地图归一化坐标）：由开局阵营决定（蓝方右下 / 红方左上镜像）
LANE_DIR_BLUE = (0.72, 0.82)
LANE_DIR_RED = (0.28, 0.18)
# 己方泉水方向（小地图坐标）：蓝方左下 / 红方右上（镜像）
FOUNTAIN_DIR_BLUE = (0.12, 0.88)
FOUNTAIN_DIR_RED = (0.88, 0.12)
# 低血量撤退（用户规则）：HP < 20% 时朝泉水移动或安全回城
LOW_HP_FRAC = 0.20
DANGER_FRAC = 0.50            # 身边有危险：敌方英雄/兵 < 0.5 屏宽
RECALL_THROTTLE_S = 20.0      # 回城节流（避免反复点）
# 塔规避：屏幕存在敌方塔且无我方小兵时，移动方向朝塔则偏转远离
TURRET_SAFE_FRAC = 0.55       # 塔在屏幕内时与移动目标方向冲突判定阈值
TURRET_THREAT_FRAC = 0.45     # 塔威胁距离：塔中心距屏幕中心 < 0.45 屏宽才算威胁
TURRET_ESCAPE_FRAC = 0.40     # 塔中心 < 0.40 屏宽（≈塔射程内）→ 已被塔打，立即反向逃离
ATTACK_THROTTLE_S = 1.0       # 普攻节流（攻击间隔）
MELEE_FRAC = 0.20             # 贴身距离：敌人/兵 < 0.20 屏宽且技能冷却 → 普攻
# ---- v2.5（用户示范局讲解新增规则）----
HOOK_BLOCK_FRAC = 0.05        # 钩子"粗直线"半宽：路径两侧各 0.05 屏宽内有小兵/野怪 → 不钩（会被挡）
RESTORE_THROTTLE_S = 10.0     # 恢复键节流
LOW_RESOURCE_FRAC = 0.80      # HP 或 MP < 80% 且安全 → 按恢复键（用户规则）
CHASE_BREAK_FRAC = 0.50       # 追击中断：HP < 50% 时停止追击（用户规则：血量降半停止追击撤退）
ALLY_TOWER_RECALL_FRAC = 0.35  # 残血回城：自家塔中心 < 0.35 屏宽才安全回城（塔下回城）
SAFE_CLEAR_FRAC = 0.55        # 小地图安全判断：附近红点距离 > 0.55 视为安全（可先清兵再回城）


def _debounced(action: dict, now: float, cooldowns: dict) -> dict:
    """技能释放后 50ms 内不再重复移动（防抖）；技能本身不受防抖限制。"""
    if now - float(cooldowns.get("skill", 0.0)) < SKILL_DEBOUNCE_S:
        return {"type": "none", "reason": "skill_debounce"}
    return action


def _nearest_walkable(grid, p, radius=3):
    """网格中距离 p 半径 radius 内的第一个可走点，找不到返回 None。"""
    n = grid.shape[0]
    for r in range(1, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = p[0] + dy, p[1] + dx
                if 0 <= y < n and 0 <= x < n and grid[y, x] > 0:
                    return (y, x)
    return None


def _hook_path_blocked(target, blockers, half_w, dist_width, aspect):
    """钩子遮挡检查（用户规则 007）：敌人连线视为"粗直线"，
    若线上有敌方小兵/野怪（blockers）且比敌人更近 -> 钩子会被挡，不释放。

    target: [cx, cy, w, h] 屏幕归一化（我方在屏幕中心 (0.5, 0.5)）
    判断：blocker 到线段（中心->target）的距离 < half_w 且 blocker 距离 < target 距离
    """
    import math
    tx, ty = target[0], target[1]
    dx, dy = tx - 0.5, (ty - 0.5) * aspect
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-4:
        return False
    for b in blockers:
        bx, by = b[0], b[1]
        # 点到线段距离（参数 t 投影）
        px, py = bx - 0.5, (by - 0.5) * aspect
        t = (px * dx + py * dy) / (seg_len * seg_len)
        if t < 0.0 or t > 1.0:
            continue
        # 完整垂直距离（x/y 折算后）
        proj_x = 0.5 + dx * t
        proj_y = 0.5 + dy * t
        perp = math.hypot(bx - proj_x, (by - proj_y) * aspect)
        b_dist = math.hypot(bx - 0.5, (by - 0.5) * aspect)
        if perp < half_w and b_dist < seg_len * 0.98:
            return True
    return False


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
    enemies, minions, turrets, allies, ally_minions, monsters = [], [], [], [], [], []
    ally_turrets = []
    for u in units:
        cls = str(u.get("cls", ""))
        scr = u.get("screen") or [0.5, 0.5, 0.0, 0.0]
        cx, cy = scr[0], scr[1]
        # v2.8 UI 区过滤：右上角头像区 (x>0.72, y<0.26) 的塔/英雄检测是 UI 误检
        if cls in ("enemy_turret", "ally_turret", "enemy_hero", "ally_hero") \
                and cx > 0.72 and cy < 0.26:
            continue
        # 小地图区 (x<0.18, y<0.32) 的检测也可能含小地图图标
        if cls in ("enemy_turret", "ally_turret") and cx < 0.18 and cy < 0.32:
            continue
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
        elif cls == "neutral_monster":
            monsters.append(scr)
        elif cls == "ally_turret":
            ally_turrets.append(scr)

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

    # ---- 0.5) 低血量撤退（用户规则）：HP<20% 回自家塔下再回城 ----
    hp = float((state_dict.get("ui") or {}).get("hp") or 1.0)
    if hp < LOW_HP_FRAC:
        dangerous = any(dist_width(s[0], s[1]) < DANGER_FRAC
                        for s in (enemies + minions))
        camp = cooldowns.get("camp") or "blue"
        fx, fy = FOUNTAIN_DIR_BLUE if camp == "blue" else FOUNTAIN_DIR_RED
        # v2.5：残血回城要在自家塔下（自家塔中心 < 0.35 屏宽）才安全回城
        nt_ally = nearest(ally_turrets)
        under_ally_tower = (nt_ally is not None
                            and dist_width(nt_ally[0], nt_ally[1]) < ALLY_TOWER_RECALL_FRAC)
        if not dangerous and under_ally_tower and ready("recall_t", RECALL_THROTTLE_S):
            cooldowns["recall_t"] = now
            return {"type": "recall", "reason": "low_hp_safe_recall"}
        # 不在塔下 -> 朝自家塔走（无塔可见则朝泉水）
        if nt_ally is not None:
            tx, ty = nt_ally[0], nt_ally[1]
            return {"type": "move", "theta": math.atan2(-(ty - 0.5) * aspect, tx - 0.5),
                    "r": 1.0, "duration_ms": MOVE_DURATION_MS,
                    "reason": "retreat_to_ally_tower"}
        return {"type": "move", "theta": math.atan2(-(fy - 0.5) * aspect, fx - 0.5),
                "r": 1.0, "duration_ms": MOVE_DURATION_MS,
                "reason": "retreat_low_hp"}

    # ---- 0.6) 恢复键（用户规则）：HP 或 MP < 80% 且身边安全 → 按恢复键 ----
    ui = state_dict.get("ui") or {}
    mp = float(ui.get("mp") or 1.0)
    if (hp < LOW_RESOURCE_FRAC or mp < LOW_RESOURCE_FRAC) \
            and ready("restore_t", RESTORE_THROTTLE_S):
        near_danger = any(dist_width(s[0], s[1]) < DANGER_FRAC
                          for s in (enemies + minions + monsters))
        if not near_danger:
            cooldowns["restore_t"] = now
            return {"type": "restore", "reason": "low_resource_safe_restore"}

    # ---- 1) 塔规避：敌方塔在近处可见且无我方小兵(ally_minion) → 不进入塔攻击范围 ----
    nt = nearest(turrets)
    turret_threat = bool(turrets) and not ally_minions and nt is not None \
        and dist_width(nt[0], nt[1]) < TURRET_THREAT_FRAC
    if turret_threat:
        tx, ty = nt[0], nt[1]
        # 已在塔攻击范围内（塔中心 < ESCAPE_FRAC）→ 立即反向逃离（用户反馈：被塔打不知道出去）
        if dist_width(tx, ty) < TURRET_ESCAPE_FRAC:
            away_theta = math.atan2(-(ty - 0.5) * aspect, -(tx - 0.5))
            return {"type": "move", "theta": away_theta, "r": 1.0,
                    "duration_ms": MOVE_DURATION_MS, "reason": "escape_turret"}
        ne = nearest(enemies)
        if ne is None:
            # 无敌人时也不朝塔方向走（朝远离塔方向）
            away_theta = math.atan2(-(ty - 0.5) * aspect, -(tx - 0.5))
            return {"type": "move", "theta": away_theta, "r": MOVE_R,
                    "duration_ms": MOVE_DURATION_MS, "reason": "avoid_turret"}
        # 有敌人时：若敌人与塔同向且塔威胁，仍可钩（钩子不进场），移动方向保持但标记
        cooldowns["turret_threat"] = 1.0
    else:
        cooldowns["turret_threat"] = 0.0

    # ---- 2) 二技能：敌方英雄在钩子范围内（v2.5 增加：钩子路径被小兵/野怪挡住则不钩）----
    skill_states = (state_dict.get("ui") or {}).get("skill_states") or {}
    if enemies:
        ne = nearest(enemies)
        d = dist_width(ne[0], ne[1])
        if d <= SKILL2_RANGE_FRAC and ready("skill2_t", SKILL2_THROTTLE_S):
            # 技能解锁检查（v2.5）：未解锁（灰暗）不释放
            s2 = skill_states.get("2") or {}
            if s2.get("unlocked") is False:
                pass  # 未解锁 -> 跳过钩子，落普攻/移动
            else:
                # 钩子遮挡检查（用户规则 007）：敌人连线"粗直线"上有敌兵/野怪 -> 不钩
                blocked = _hook_path_blocked(
                    ne, minions + monsters, HOOK_BLOCK_FRAC, dist_width, aspect)
                if not blocked:
                    cooldowns["skill2_t"] = now
                    cooldowns["skill"] = now
                    cooldowns["hook_pending"] = now
                    cooldowns["hook_anchor_dist"] = d
                    return {"type": "skill", "id": 2, "mode": "tap",
                            "reason": "enemy_in_skill2_range"}
                cooldowns["hook_blocked"] = 1.0
            cooldowns["hook_blocked"] = cooldowns.get("hook_blocked", 0.0)
        else:
            cooldowns["hook_blocked"] = 0.0

    # ---- 3) 一技能：不在大招生效期 且 敌人/敌兵在身边（v2.5：未解锁则跳过，落普攻）----
    in_ult = now - float(cooldowns.get("skill3_t", 0.0)) <= SKILL3_ACTIVE_S
    s1 = skill_states.get("1") or {}
    if not in_ult and s1.get("unlocked") is not False \
            and ready("skill1_t", SKILL1_THROTTLE_S):
        ne = nearest(enemies)
        ne_m = nearest(minions)
        d_e = dist_width(ne[0], ne[1]) if ne else float("inf")
        d_m = dist_width(ne_m[0], ne_m[1]) if ne_m else float("inf")
        if min(d_e, d_m) < NEAR_FRAC:
            cooldowns["skill1_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 1, "mode": "tap",
                    "reason": "enemy_or_minion_near"}

    # ---- 3.5) 普攻：敌人/敌兵贴身（< MELEE_FRAC）且技能不可用/冷却时 → 攻击键 ----
    # （用户反馈：面对敌方英雄不攻击、被动挨打 → 贴身时用普攻）
    if ready("attack_t", ATTACK_THROTTLE_S):
        ne = nearest(enemies)
        ne_m = nearest(minions)
        d_e = dist_width(ne[0], ne[1]) if ne else float("inf")
        d_m = dist_width(ne_m[0], ne_m[1]) if ne_m else float("inf")
        if min(d_e, d_m) < MELEE_FRAC:
            cooldowns["attack_t"] = now
            return {"type": "attack", "priority": "free", "reason": "melee_attack"}

    # ---- 4) 移动：持续拖动 ----
    # v2.7 目标滞回：目标频繁跳变（蓝点噪声）导致乱转 -> 目标需稳定 HOLD_S 秒才切换
    TARGET_HOLD_S = 1.5
    if enemies and hp < CHASE_BREAK_FRAC:
        camp = cooldowns.get("camp") or "blue"
        fx, fy = FOUNTAIN_DIR_BLUE if camp == "blue" else FOUNTAIN_DIR_RED
        target = (fx, fy)
        reason = "stop_chase_low_hp_retreat"
        theta = math.atan2(-(fy - 0.5) * aspect, fx - 0.5)
        if now - float(cooldowns.get("skill2_t", 0.0)) < HOOK_FLIGHT_S:
            return {"type": "none", "reason": "hook_flight"}
        return {"type": "move", "theta": theta, "r": 1.0,
                "duration_ms": MOVE_DURATION_MS, "reason": reason}
    target = None
    reason = None
    if enemies:
        ne = nearest(enemies)
        target = (ne[0], ne[1])
        reason = "chase_enemy"
    else:
        # 移动决策（用户规则：通过小地图观察敌我位置）：
        #   1) 跟随队友：屏幕 ally_hero 优先；否则小地图蓝点中离发育路最近的（射手优先）
        #   2) 无队友：朝小地图红点（支援战斗）
        #   3) 无红点：朝发育路方向（按阵营镜像）
        camp = cooldowns.get("camp") or "blue"
        lane = LANE_DIR_BLUE if camp == "blue" else LANE_DIR_RED
        mm = state_dict.get("minimap") or {}
        mm_blue, mm_red = [], []
        if mm.get("found"):
            mm_blue = (mm.get("dots") or {}).get("blue") or []
            mm_red = (mm.get("dots") or {}).get("red") or []

        # v2.7 开局保护：对局确认后前 15 秒只朝发育路走
        # （开局红点/蓝点检测不稳，乱支援会导致泉水乱转）
        match_t = float(cooldowns.get("match_start_t", 0.0))
        in_opening = match_t > 0 and now - match_t < 15.0
        na = nearest(allies)
        if na is not None and not in_opening:
            target = (na[0], na[1])
            reason = "follow_ally"
        elif mm_blue and not in_opening:
            # 射手优先：离发育路方向最近的蓝点（射手常在发育路）
            # v2.7：不再排除"己方点"（离圆心最近≠己方，会误判导致目标错乱），
            # 目标跳变由滞回处理
            lx, ly = lane
            target = min(mm_blue, key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2)
            reason = "follow_ally_minimap"
        elif mm_red and not in_opening:
            lx = sum(p[0] for p in mm_red) / len(mm_red)
            ly = sum(p[1] for p in mm_red) / len(mm_red)
            target = (lx, ly)
            reason = "support_red_centroid"
        else:
            lx, ly = lane
            target = (lx, ly)
            reason = "lane_develop"
    if target is None:
        return {"type": "none", "reason": "no_target"}
    # v2.7 目标滞回：蓝点噪声导致目标每帧跳变 -> 目标需稳定 HOLD_S 秒才切换
    # （仅对小地图类目标生效；chase_enemy 追敌人不滞回，避免追丢）
    if reason in ("follow_ally_minimap", "support_red_centroid", "lane_develop"):
        prev = cooldowns.get("nav_target")
        prev_t = float(cooldowns.get("nav_target_t", 0.0))
        if prev is not None:
            dist = math.hypot(target[0] - prev[0], target[1] - prev[1])
            if dist > 0.12 and now - prev_t < TARGET_HOLD_S:
                # 目标跳变但未到切换期 -> 维持旧目标（刷新时间戳，避免永久 hold）
                target = tuple(prev)
                reason = f"{reason}_hold"
                cooldowns["nav_target"] = tuple(target)
                cooldowns["nav_target_t"] = now
            else:
                cooldowns["nav_target"] = tuple(target)
                cooldowns["nav_target_t"] = now
        else:
            cooldowns["nav_target"] = tuple(target)
            cooldowns["nav_target_t"] = now
    # 钩子飞行期停止移动（等勾中结果，防自己走近误判连招）
    if now - float(cooldowns.get("skill2_t", 0.0)) < HOOK_FLIGHT_S:
        return {"type": "none", "reason": "hook_flight"}
    theta = math.atan2(-(target[1] - 0.5) * aspect, target[0] - 0.5)
    # ---- 小地图寻路修正（v2.6）：目标方向被墙体挡 -> 用模板 A* 的第一步方向 ----
    mm = state_dict.get("minimap") or {}
    # 仅当目标是小地图坐标（lane/support/follow_minimap/retreat）时修正；
    # follow_ally（屏幕目标）与 chase_enemy 不做寻路（屏幕近距目标直接走）
    if mm.get("found") and reason in ("lane_develop", "follow_ally_minimap",
                                      "support_red_centroid", "retreat_low_hp"):
        from wzry.vision.terrain import astar_path
        from wzry.vision.terrain_map import load_terrain
        try:
            tgrid = load_terrain()
            n = tgrid.shape[0]
            # 己方位置（v2.8：绿色点=自己；退化蓝点）
            self_p = None
            greens = (mm.get("dots") or {}).get("green") or []
            if greens:
                self_p = min(greens, key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
            else:
                blues = (mm.get("dots") or {}).get("blue") or []
                if blues:
                    self_p = min(blues, key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
            if self_p and 0.0 <= target[0] <= 1.0 and 0.0 <= target[1] <= 1.0:
                sg = (min(n - 1, max(0, int(self_p[1] * n))),
                      min(n - 1, max(0, int(self_p[0] * n))))
                tg = (min(n - 1, max(0, int(target[1] * n))),
                      min(n - 1, max(0, int(target[0] * n))))
                # 起点/终点不可走时找最近可走格（半径 3 内）
                sg = _nearest_walkable(tgrid, sg, 3) or sg
                tg = _nearest_walkable(tgrid, tg, 3) or tg
                if tgrid[sg] > 0 and tgrid[tg] > 0:
                    path = astar_path(tgrid, sg, tg)
                    if path and len(path) >= 2:
                        # 第一步方向（网格坐标差 -> 屏幕方向：网格y向下=屏幕y向下）
                        dgy = path[1][0] - path[0][0]
                        dgx = path[1][1] - path[0][1]
                        if dgy or dgx:
                            theta = math.atan2(dgy, dgx)
                            reason = f"{reason}_path"
        except Exception:
            pass
    # 塔规避：若移动方向朝向威胁塔 → 反向逃离（比垂直绕行更坚决，用户反馈被塔打要赶紧出去）
    if turret_threat and cooldowns.get("turret_threat"):
        tx, ty = nt[0], nt[1]
        t_theta = math.atan2(-(ty - 0.5) * aspect, tx - 0.5)
        diff = abs(((theta - t_theta + math.pi) % (2 * math.pi)) - math.pi)
        if diff < math.pi / 4:  # 目标方向与塔同向 -> 直接反向跑出塔范围
            away_theta = t_theta + math.pi
            return {"type": "move", "theta": away_theta, "r": 1.0,
                    "duration_ms": MOVE_DURATION_MS, "reason": "escape_turret"}
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
    elif t == "recall":
        cooldowns["recall_t"] = now
        cooldowns["skill"] = now
    elif t == "attack":
        cooldowns["attack_t"] = now
    elif t == "restore":
        cooldowns["restore_t"] = now
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
    elif t == "recall":
        ex.recall()
    elif t == "attack":
        ex.attack(str(action.get("priority", "free")))
    elif t == "restore":
        ex.restore()
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
    if t == "recall":
        return f"回城 [{reason}]"
    if t == "restore":
        return f"恢复键 [{reason}]"
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

    # v2.8：rotate_s 调大到 30 分钟（避免对局中轮转导致"无帧"停顿）
    cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy", rotate_s=1800)
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
        from wzry.vision.minimap_tracker import MinimapTracker
        from wzry.vision.terrain import DEFAULT_BOX
        return MinimapTracker(prior_center=mm_prior, box_prior=DEFAULT_BOX)

    tracker = None
    # 撞墙感知（v2.6：轮盘拖动但小地图蓝点不动 -> 绕行）
    from wzry.vision.wall_sensor import WallSensor
    wall_sensor = WallSensor()

    if do_action:
        from wzry.action.executor_v2 import ActionExecutor
        ex = ActionExecutor()
        print("动作模式: 执行（--action，真机触摸）")
    else:
        ex = None
        print("动作模式: 仅观察（默认 --no-action，不注入任何触摸；加 --action 才执行）")

    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")
    cooldowns = {"skill1_t": 0.0, "skill2_t": 0.0, "skill3_t": 0.0,
                 "summoner_t": 0.0, "recall_t": 0.0, "hp_t": 0.0,
                 "restore_t": 0.0, "attack_t": 0.0,
                 "skill": 0.0, "hook_pending": 0.0,
                 "hook_anchor_dist": 0.0, "turret_threat": 0.0,
                 "hook_blocked": 0.0, "match_start_t": 0.0,
                 "roster": None, "roster_pending": 0.0}
    # 技能按钮像素坐标（read_ui 用）
    with open(ROOT / "configs" / "calibration_absolute.json", encoding="utf-8") as _f:
        _pts = json.load(_f)["points"]
    skills_pts = {1: _pts["skill1"], 2: _pts["skill2"], 3: _pts["skill3"]}

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
    # v2.7 性能优化：YOLO 降频（每 3 帧检测一次，结果缓存复用）
    yolo_every = 3
    yolo_count = 0
    cached_dets = []
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
            # v2.7 开局保护计时（对局确认时刻）
            if not cooldowns.get("match_start_t"):
                cooldowns["match_start_t"] = now

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

            # ---- 开局阵容识别（v2.10：modlens 读选英雄界面文字，含钟馗侧=我方）----
            # 用后台线程避免阻塞主循环；结果存入 cooldowns["roster"]
            if not cooldowns.get("roster") and not cooldowns.get("roster_pending"):
                cooldowns["roster_pending"] = now
                import threading as _th
                _frame_cp = frame.copy()

                def _roster_worker(fr):
                    try:
                        import subprocess as _sp
                        import sys as _sys
                        tmp = ROOT / "temp" / "roster_live.png"
                        import cv2 as _cv2
                        _cv2.imwrite(str(tmp), fr)
                        prompt = ('这是王者荣耀选英雄界面。请列出上方区域(y0-360)的所有英雄名和'
                                  '下方区域(y360-720)的所有英雄名，用JSON格式输出：'
                                  '{"upper": ["英雄名"...], "lower": ["英雄名"...]}。'
                                  '只输出英雄名（去掉皮肤名），格式必须严格是JSON。')
                        r = _sp.run(
                            [_sys.executable, "-X", "utf8",
                             str(ROOT / "scripts" / "train" / "modlens_ask.py"),
                             str(tmp), prompt],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=180)
                        if r.returncode != 0:
                            print(f"[roster] modlens 失败: {r.stderr[:150]}")
                            return
                        txt = r.stdout.strip()
                        start, end = txt.find("{"), txt.rfind("}")
                        if start < 0 or end < 0:
                            print(f"[roster] 解析失败: {txt[:150]}")
                            return
                        import json as _json
                        data = _json.loads(txt[start:end + 1])
                        upper, lower = data.get("upper", []), data.get("lower", [])
                        if "钟馗" in upper:
                            ally, enemy = upper, lower
                        else:
                            ally, enemy = lower, upper
                        cooldowns["roster"] = {"ally": ally, "enemy": enemy,
                                               "self_hero": "钟馗"}
                        print(f"[roster] 我方 {ally} | 敌方 {enemy}")
                    except Exception as e:
                        print(f"[roster] 异常: {e}")
                    finally:
                        cooldowns["roster_pending"] = 0.0

                _th.Thread(target=_roster_worker, args=(_frame_cp,), daemon=True).start()

            # ---- 感知（v2.7：YOLO 降频，缓存复用）----
            yolo_count += 1
            if yolo_count >= yolo_every:
                yolo_count = 0
                cached_dets = det.detect(frame)
            dets = cached_dets
            st = build_state(frame, dets, phase.value, minimap={
                "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
                "dots": mm["dots"], "towers": mm["towers"],
            }, frame_id=frame_id)
            frame_id += 1
            st.t = time.time()  # 决策时刻
            state_dict = st.to_dict()
            # 低血量决策：读取 HP/MP（左下角血条，低频 0.5s 一次）
            if now - float(cooldowns.get("hp_t", 0.0)) >= 0.5:
                from wzry.vision.ui_reader import read_ui
                ui_res = read_ui(frame, skills_pts)
                cooldowns["hp_t"] = now
                if ui_res["hp_bar"]:
                    state_dict.setdefault("ui", {})["hp"] = ui_res["hp_bar"][-1]
                if ui_res["mp_bar"]:
                    state_dict.setdefault("ui", {})["mp"] = ui_res["mp_bar"][-1]
                if ui_res["skill_states"]:
                    state_dict.setdefault("ui", {})["skill_states"] = ui_res["skill_states"]
            n_ticks += 1
            infer_sum += det.last_infer_ms

            # ---- 规则决策 ----
            action = decide(state_dict, cooldowns)
            sig = (action.get("type"), action.get("id"))
            if sig != last_sig:
                print(f"[{datetime.now():%H:%M:%S}] [决策] {format_decision(action)}")
                last_sig = sig

            # ---- 撞墙感知（v2.6）：移动中自己位置不动 -> 绕行 ----
            if action.get("type") == "move":
                # v2.8：己方位置 = 小地图绿色点（用户语义：绿圈=自己）
                hero_pos = None
                if mm.get("found"):
                    greens = (mm.get("dots") or {}).get("green") or []
                    if greens:
                        hero_pos = min(greens, key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
                    else:
                        # 绿色点未检出时退化为蓝色点（队友近似，宽容）
                        blues = (mm.get("dots") or {}).get("blue") or []
                        if blues:
                            hero_pos = min(blues, key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
                wall_hit = wall_sensor.update(now, True, hero_pos)
                if wall_hit:
                    action = wall_sensor.avoid_action(float(action.get("theta", 0.0)))
                    print(f"[{datetime.now():%H:%M:%S}] [撞墙] 位置未动 -> 绕行")
                    sig = ("move", None)
            else:
                wall_sensor.update(now, False, None)

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
                # v2.10：阵容写入状态流（决策层可读）
                if cooldowns.get("roster"):
                    state_dict.setdefault("extra", {})["roster"] = cooldowns["roster"]
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
