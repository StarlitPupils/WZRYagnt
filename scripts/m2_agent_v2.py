# -*- coding: utf-8 -*-
# m2_agent_v2 line 2
#鏋舵瀯垀10Hz 鎰熺煡寰幆夛細
#    scrcpy 娴侊紙30fps夆啋 MatchStateMachine堝灞鐘舵佹満      鈫纭鍒讹紙in_match + 忓湴鍥捐繛缁2 甯璁級
#      鈫鎰熺煡圷OLO 11 绫绘娴+ MinimapTracker      鈫decide() 瑙勫垯鍐崇瓥堢函鍑芥暟屾棤 IO屼究浜庢祴璇鎹瓥鐣綉缁滐級
#      鈫ActionExecutor 鎵-action 鎵嶇湡姝ｈ鎽革紝榛樿 --no-action 鍙瀵燂級

#瑙勫垯浼樺厛绾紙v0 绠鍗曚絾瀹夊叏岃 decide()夛細
#  1. 閽瓙鍙懡涓紙妫娴嬪埌 hook_aim 鎴鏁屾柟鑻遍泟鍦2 鎶鑳借窛绂诲唴     鈫skill_cast(2, 'tap')屽喎鍗磋妭娴> 3s  2. 鏁屼汉杩戯紙enemy_hero 涓績璺濆睆骞曚腑蹇< 0.25 灞忓     鈫skill_cast(1, 'tap')岃妭娴1.5s  3. 鏁屼汉璐磋韩 0.20 灞忓変笖鎶鑳戒笉鍙敤 鈫鏅敾堟敾鍑婚敭夛紝鑺傛祦 1s  4. 鏁屼汉杩鈫move(鏈濆悜鏈杩戞晫浜 r=0.8, 400ms)  5. 鏃犳晫浜鈫鏈濆叺绾挎柟鍚戠鍔紙忓湴鍥剧孩鐐硅川蹇冩柟鍚戯級涙棤绾偣鍒idle銆  闃叉姈氫换浣曟妧鑳介噴鏀惧悗 50ms 鍐呬笉鍐嶄笅鍙戠鍔  濉旓細琚鏀诲嚮堝涓績 < 0.40 灞忓涓旀棤鎴戞柟忓叺夆啋 绔嬪嵆鍙嶅悜閫冪
#鐢硶    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120              # 鍙瀵燂紙榛樿屽畨鍏紑鍏筹級
#    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action     # 鐪熸満鎵
#    venv\\Scripts\\python.exe scripts\\m2_agent_v2.py --seconds 120 --action --save   # 鎵 + 瀵瑰眬浼氳瘽閲囬泦
#    """decoded docstring."""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# decision params
SKILL1_THROTTLE_S = 1.5
SUMMONER_THROTTLE_S = 30.0
SKILL3_ACTIVE_S = 3.5
SKILL_DEBOUNCE_S = 0.05
SKILL2_THROTTLE_S = 3.0      # hook throttle 3s (anti-AFK)
SKILL2_RANGE_FRAC = 0.45
SKILL1_RANGE_FRAC = 0.32   # v3.5 一技能(近身消耗)范围: 勾未中时只在一技能范围内用一技能
SKILL_HP_MIN = 0.60
HOOK_FLIGHT_S = 0.5           # 閽瓙椋炶鏈燂細閲婃斁鍚0.5s 鍐呭仠姝鍔紙绛夊嬀涓粨鏋滐紝闃茶嚜宸辫蛋杩戣鍒級
# throttle consts
NEAR_FRAC = 0.25
HOOK_CONFIRM_S = 1.0
HOOK_DIST_SHRINK = 0.72
MOVE_R = 0.8
MOVE_DURATION_MS = 450
LANE_DIR_BLUE = (0.75, 0.72)   # v2.91: 发育路中心(蓝方右下), 旧(0.5,0.5)=地图中心=河道! 用户: 别待河道
LANE_DIR_RED = (0.25, 0.28)    # v2.91: 发育路中心(红方左上镜像)
DANGER_FRAC = 0.45
# 宸辨柟娉夋按鏂瑰悜堝皬鍦板浘鍧愭爣夛細钃濇柟宸笅 / 绾柟鍙充笂堥暅鍍忥級
FOUNTAIN_DIR_BLUE = (0.12, 0.88)
FOUNTAIN_DIR_RED = (0.88, 0.12)
# 浣庤閲忔挙閫堢敤鎴疯鍒v2.19 瀛範: 鐢埛 HP<40%+绾偣杩= 鎾 75%夆斺0% 鍗虫挙
LOW_HP_FRAC = 0.40
TURRET_ESCAPE_FRAC = 0.40
ATTACK_THROTTLE_S = 0.8
MELEE_FRAC = 0.30
HOOK_BLOCK_FRAC = 0.05
RESTORE_THROTTLE_S = 60.0
LOW_RESOURCE_FRAC = 0.80
CHASE_BREAK_FRAC = 0.50
ALLY_TOWER_RECALL_FRAC = 0.35
SAFE_CLEAR_FRAC = 0.55
def _debounced(action: dict, now: float, cooldowns: dict) -> dict:
    """decoded docstring."""
    if now - float(cooldowns.get("skill", 0.0)) < SKILL_DEBOUNCE_S:
        return {"type": "none", "reason": "skill_debounce"}
    return action


def _mm_self(state_dict):
    """decoded docstring."""
    mm = state_dict.get("minimap") or {}
    dots = (mm.get("dots") or {}) if mm.get("found") else {}
    greens = dots.get("green") or []
    # v2.89: 无自点(未追踪到)返回 None, 调用方不作为红点距离基准(旧(0.5,0.5)兜底
    # -> 泉水期任意中点红点<0.15 触发空放技能)
    return (greens[0][0], greens[0][1]) if greens else None


def _away_map(state_dict, tx, ty, k=0.12):
    """decoded docstring."""
    gx, gy = _mm_self(state_dict) or (0.5, 0.5)   # v2.89 None兜底(仅方位辅助)
    d = math.hypot(gx - tx, gy - ty) or 1e-6
    return (max(0.02, min(0.98, gx + (gx - tx) / d * k)),
            max(0.02, min(0.98, gy + (gy - ty) / d * k)))


def _nearest_walkable(grid, p, radius=3):
    """decoded docstring."""
    n = grid.shape[0]
    for r in range(1, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = p[0] + dy, p[1] + dx
                if 0 <= y < n and 0 <= x < n and grid[y, x] > 0:
                    return (y, x)
    return None


def _hook_path_blocked(target, blockers, half_w, dist_width, aspect):
    """decoded docstring."""
#    target: [cx, cy, w, h] 灞忓箷褰掍竴鍖栵紙鎴戞柟鍦睆骞曚腑蹇(0.5, 0.5)    鍒柇歜locker 鍒扮嚎娈碉紙涓績->target夌殑璺濈 < half_w 涓blocker 璺濈 < target 璺濈
    """decoded docstring."""
    import math
    tx, ty = target[0], target[1]
    dx, dy = tx - 0.5, (ty - 0.5) * aspect
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-4:
        return False
    for b in blockers:
        bx, by = b[0], b[1]
        # 鐐瑰埌绾挎璺濈堝弬鏁t 鎶曞奖        px, py = bx - 0.5, (by - 0.5) * aspect
        t = (px * dx + py * dy) / (seg_len * seg_len)
        if t < 0.0 or t > 1.0:
            continue
        # 瀹屾暣鍨傜洿璺濈坸/y 鎶樼畻鍚庯級
        proj_x = 0.5 + dx * t
        proj_y = 0.5 + dy * t
        perp = math.hypot(bx - proj_x, (by - proj_y) * aspect)
        b_dist = math.hypot(bx - 0.5, (by - 0.5) * aspect)
        if perp < half_w and b_dist < seg_len * 0.98:
            return True
    return False


def decide(state_dict: dict, cooldowns: dict) -> dict:
    """decoded docstring."""
    # (comment)
#      0. 鍕句腑杩炴嫑氫簩鎶鑳介噴鏀惧悗绐楀彛鍐呮晫浜鸿鎷夎繎 鈫鍙敜甯堟妧鑳+ 涓夋妧鑳      1. 濉旇閬匡細鏁屾柟濉斿彲瑙佷笖鏃犳垜鏂瑰皬鍏鈫绉诲姩閬垮紑濉旀敾鍑昏寖鍥达紙鎶鑳戒笉鍙楅檺      2. 浜屾妧鑳斤細鏁屾柟鑻遍泟鍦簩鎶鑳借寖鍥村唴 鈫閽瓙
#      3. 涓鎶鑳斤細涓嶅湪澶嫑鐢熸晥鏈涓鏁屼汉/鏁屽叺鍦韩杈鈫涓鎶鑳      4. 绉诲姩氭湁鏁屼汉 鈫鏈濇渶杩戞晫浜烘寔缁嫋鍔紱鏃犳晫浜鈫甯彂鑲茶矾勬墜
#         堣窡闅忔渶杩ally_hero屽惁鍒欐湞鍙戣偛璺柟鍚戯級

#    璁板繂坈ooldowns 鎵睍瀛楁夛細
#      skill2_t / skill1_t / skill3_t / summoner_t  鏈杩戦噴鏀炬椂闂      hook_anchor_dist  浜屾妧鑳介噴鏀炬椂鏈杩戞晫浜鸿窛绂伙紙鍕句腑鍒畾鍩哄噯      hook_pending      浜屾妧鑳介噴鏀炬椂鍒伙紙鍕句腑纭绐楄捣鐐癸級
    """decoded docstring."""
    now = float(state_dict.get("t") or 0.0)
    w, h = state_dict.get("screen_size") or (1280.0, 720.0)
    aspect = (float(h) / float(w)) if w else 0.5625

    # input: extra enemy bars / self pos
    extra = state_dict.get("extra") or {}
    enemy_bars = extra.get("enemy_bars") or []
    roster = extra.get("roster") or cooldowns.get("roster")
    hook_signal = extra.get("hook_signal") or []

    units = state_dict.get("units") or []
    enemies, minions, turrets, allies, ally_minions, monsters = [], [], [], [], [], []
    ally_turrets = []
    selfs = []                      # v2.45 self collection
    for u in units:
        cls = str(u.get("cls", ""))
        scr = u.get("screen") or [0.5, 0.5, 0.0, 0.0]
        cx, cy = scr[0], scr[1]
        # v2.8 UI 鍖鸿繃婊細鍙充笂瑙掑鍍忓尯 (x>0.72, y<0.20) 鐨勫/鑻遍泟妫娴嬫槸 UI 璇
        if cls in ("enemy_turret", "ally_turret", "enemy_hero", "ally_hero") \
                and cx > 0.72 and cy < 0.20:
            continue
        # (comment)
        if cls in ("enemy_turret", "ally_turret") and cx < 0.18 and cy < 0.32:
            continue
        if cls == "self":
            pass   # v2.46 鍏睆鑷繁涓嶆娴涓嶆樉绀 鑷繁蹇呭湪灞忓箷涓, 浠皬鍦板浘缁跨幆涓哄噯
        elif cls == "enemy_hero":
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

#    # v2.45 鑷繁=鍏満鍞竴: 澶鑷繁"妗-> 绂诲睆骞曚腑蹇冩渶杩戣鐪熸垜, 鍏朵綑杞洖鎴戣嫳(闃熷弸璇爣)
    if len(selfs) > 1:
        selfs.sort(key=lambda s: (s[0] - 0.5) ** 2 + (s[1] - 0.5) ** 2)
        # (comment)
        selfs = selfs[:1]
    elif selfs:
        pass
#    # 鑷繁涓庢垜鑻遍噸鍙闃熷弸璐磋劯) -> 浠鑷繁"涓哄噯, 鍘绘帀涓庡叾閲嶅彔鐨勬垜鑻    if selfs:
        sx, sy = selfs[0][0], selfs[0][1]
        allies = [a for a in allies if math.hypot(a[0] - sx, a[1] - sy) > 0.04]

    # v2.60 敌英信任
    _mmp_yellow = ((state_dict.get("minimap") or {}).get("dots") or {}).get("yellow") or []
    # (comment)
    _ally_bars = extra.get("ally_bars") or []
    _ally_bars = extra.get("ally_bars") or []
    if allies and _ally_bars:
        allies = [a for a in allies if any(
            math.hypot(b[0] / w - float(a[0]), b[1] / h - float(a[1])) < 0.20
            for b in _ally_bars)]

    def dist_width(cx, cy):
        return math.hypot(cx - 0.5, (cy - 0.5) * aspect)

    def nearest(lst):
        if not lst:
            return None
        return min(lst, key=lambda s: dist_width(s[0], s[1]))

    def ready(key, thr):
        return now - float(cooldowns.get(key, 0.0)) > thr

    # ---- v2.72 dominant skills; v2.97 用户规则(铁): 小地图红点**禁止**触发技能/出钩
    #      (红点闪烁位置不准 -> 乱出钩); 只有小地图之外=屏幕上 敌英框/敌红条 才释放。
    #      出钩(skill2)额外要求: 屏幕敌英在二技能范围内(SKILL2_RANGE_FRAC) ----
    _mmr_pts0 = ((state_dict.get("minimap") or {}).get("dots") or {}).get("red") or []
    _mm_own0 = _mm_self(state_dict)
    # v2.89 红点距离只有在自己绿点已知时才有意义(泉水/未追踪=0.5,0.5兜底曾触发空放)
    _d_red0 = (min([math.hypot(p[0] - _mm_own0[0], p[1] - _mm_own0[1]) for p in _mmr_pts0],
                   default=9.9) if _mm_own0 else 9.9)
    if _mmr_pts0:
        cooldowns["red_stable"] = float(cooldowns.get("red_stable", 0)) + 1
    else:
        cooldowns["red_stable"] = 0
    # v2.97 释放证据 = 屏幕敌英(yolo框) 或 屏幕敌红条(yolo未检出时可作辅助);
    #          v2.98 红条独证不释放技能: 暴君/红buff等野怪红条也是红条且yolo常漏检
    #          -> 无 yolo enemy_hero 框时, 红条只算"疑似敌英", 不触发任何技能/出钩
    _has_yolo_eh = bool(enemies)
    if _has_yolo_eh or enemy_bars:
        _nq0 = nearest(enemies) if enemies else None
        _dq0 = dist_width(_nq0[0], _nq0[1]) if _nq0 else 0.90
        _s_ = (state_dict.get("ui") or {}).get("skill_states") or {}
        _s3_ = _s_.get("3") or {}
        _s2_ = _s_.get("2") or {}
        _s1_ = _s_.get("1") or {}
        if ready("skill3_t", 20.0) and _s3_.get("unlocked") is not False \
                and _has_yolo_eh:
            cooldowns["skill3_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 3, "mode": "tap", "reason": "ult_near_enemy"}
        # v2.98 出钩铁律: 必须 yolo 敌英框 + 二技能范围内; 纯红条(暴君)不出钩
        if _nq0 is not None and _dq0 <= SKILL2_RANGE_FRAC \
                and ready("skill2_t", SKILL2_THROTTLE_S) and _s2_.get("unlocked") is not False:
            cooldowns["skill2_t"] = now
            cooldowns["skill"] = now
            cooldowns["hook_pending"] = now
            cooldowns["hook_anchor_dist"] = _dq0
            return {"type": "skill", "id": 2, "mode": "tap",
                    "reason": "enemy_in_skill2_range"}
        if ready("skill1_t", SKILL1_THROTTLE_S) and _s1_.get("unlocked") is not False \
                and _has_yolo_eh:
            cooldowns["skill1_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 1, "mode": "tap", "reason": "enemy_hero_near"}
        if ready("attack_t", ATTACK_THROTTLE_S) and _has_yolo_eh:
            cooldowns["attack_t"] = now
            return {"type": "attack", "priority": "free", "reason": "engage_enemy"}

    # ---- v2.13 鍥炲煄璇绘潯淇濇姢氬洖鍩庡悗 RECALL_ACTIVE_S 鍐呬笉鎵鍏朵粬鍔綔 ----
    # 堜慨澶嶏細鎭閿绉诲姩鍦洖鍩庤鏉腑鎵 -> 鍥炲煄琚墦鏂紝琛涓鐩村洖涓嶆弧    # 渚嬪氭晫浜绾潯杩戣韩堝嵄闄級鏃朵腑鏂洖鍩庯紝璧版甯稿喅绛栵紙閫冭窇/鍙嶅嚮    if now - float(cooldowns.get("recall_t", 0.0)) < RECALL_ACTIVE_S:
        ne = nearest(enemies)
        danger = ne is not None and dist_width(ne[0], ne[1]) < DANGER_FRAC
        if not danger and enemy_bars:
            for b in enemy_bars:
                bd = math.hypot(b[0] / w - 0.5, (b[1] / h - 0.5) * aspect)
                if bd < DANGER_FRAC:
                    danger = True
                    break
        if not danger:
            return {"type": "none", "reason": "recall_in_progress"}

#    # ---- 0) 鍕句腑杩炴嫑氫簩鎶鑳介噴鏀惧悗绐楀彛鍐咃紝鏁屼汉琚鎷夎繎" = 鍕惧埌浜----
    hook_pending = float(cooldowns.get("hook_pending", 0.0))
    if hook_pending > 0 and now - hook_pending <= HOOK_CONFIRM_S:
        anchor = float(cooldowns.get("hook_anchor_dist", 0.0))
        ne = nearest(enemies)
        if ne is not None and anchor > 0:
            d = dist_width(ne[0], ne[1])
            # hook hit: distance shrink (anchor-d)>0.12
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

    # ---- brainless fire (skills if enemy present) ----
    _hpq = float((state_dict.get("ui") or {}).get("hp") or 1.0)
    # v2.82 mana: measurement missing -> skill-cost estimate (0.10/skill, 12+ => recall)
    _mp_raw = (state_dict.get("ui") or {}).get("mp")
    _skill_cnt = 0
    for _sk in ("skill1_t", "skill2_t", "skill3_t"):
        _st = float(cooldowns.get(_sk, 0.0) or 0.0)
        if _st and now - _st < 90.0:
            _skill_cnt += 1
    _mp_est = max(0.0, round(1.0 - 0.10 * _skill_cnt, 2))
    if _skill_cnt >= 12 and _mp_est > 0.15:
        _mp_est = 0.15
    _mpq = float(_mp_raw) if (_mp_raw is not None and _mp_raw == _mp_raw
                              and 0.0 <= float(_mp_raw) <= 1.0) else _mp_est
    skill_states = (state_dict.get("ui") or {}).get("skill_states") or {}
    # v2.70 鏁屼汉璇佹嵁=灞忓箷鏁岃嫳妗鎴忓湴鍥剧孩鐐浠绘剰璺濈, 鐢埛璇箟: 鍑虹幇鍗虫墦)
    # v2.97 释放证据 = 屏幕敌英/敌红条(小地图红点不参与出钩与技能)
    #     v2.98 红条独证不释放: 暴君/野怪红条 yolo 漏检时算"疑似", 技能必须 yolo 敌英框
    _mmr_pts = ((state_dict.get("minimap") or {}).get("dots") or {}).get("red") or []
    _mm_own = _mm_self(state_dict)
    _d_red_near = (min([math.hypot(p[0] - _mm_own[0], p[1] - _mm_own[1]) for p in _mmr_pts],
                       default=9.9) if _mm_own else 9.9)
    _has_yolo_eh_q = bool(enemies)
    if enemies or enemy_bars:
        _nq = nearest(enemies) if enemies else None
        _dq = dist_width(_nq[0], _nq[1]) if _nq else 0.90
        _s3q = skill_states.get("3") or {}
        _s2q = skill_states.get("2") or {}
        _s1q = skill_states.get("1") or {}
        # v3.5 用户连招铁律: 勾中=接召唤师+三技能小连招; 没勾中=只用一技能消耗(不接其他)
        hook_pending = float(cooldowns.get("hook_pending", 0.0))
        hook_missed = (hook_pending > 0 and now - hook_pending > 1.2)
        if hook_pending and now - hook_pending <= 1.2 and _dq < 0.40:
            # 勾中 -> 召唤师 + 三技能 (用户明确小连招=召唤师+三技能, 不接一技能)
            combo = []
            if ready("summoner_t", 5.0):
                combo.append({"type": "summoner"})
            if ready("skill3_t", 20.0) and _s3q.get("unlocked") is not False:
                combo.append({"type": "skill", "id": 3, "mode": "tap"})
            if combo:
                for c in combo:
                    if c.get("type") == "skill":
                        cooldowns[f"skill{c['id']}_t"] = now
                    else:
                        cooldowns["summoner_t"] = now
                cooldowns["skill"] = now
                cooldowns["hook_pending"] = 0.0   # 消耗掉
                return {"type": "combo", "actions": combo, "reason": "hook_confirmed_combo"}
        # 勾未中: 只用一技能消耗(不再接三技能/召唤师/攻击), 一技能进入范围即放
        if hook_missed:
            cooldowns["hook_pending"] = 0.0
            if ready("skill1_t", SKILL1_THROTTLE_S) and _nq is not None \
                    and _dq <= SKILL1_RANGE_FRAC and _s1q.get("unlocked") is not False:
                cooldowns["skill1_t"] = now
                cooldowns["skill"] = now
                return {"type": "skill", "id": 1, "mode": "tap", "reason": "hook_missed_harass"}
        in_ult = now - float(cooldowns.get("skill3_t", 0.0)) <= 3.5
        # 常规(无钩等待): 三技能只在近敌且未勾等待时
        if ready("skill3_t", 20.0) and _s3q.get("unlocked") is not False \
                and _dq < 0.75 and _has_yolo_eh_q:
            cooldowns["skill3_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 3, "mode": "tap", "reason": "ult_near_enemy"}
        # v2.98 出钩铁律: yolo 敌英框 + 二技能范围内才出钩; 纯红条(暴君)不出钩
        if not in_ult and _nq is not None and _dq <= SKILL2_RANGE_FRAC \
                and ready("skill2_t", SKILL2_THROTTLE_S) \
                and _s2q.get("unlocked") is not False:
            cooldowns["skill2_t"] = now
            cooldowns["skill"] = now
            cooldowns["hook_pending"] = now
            cooldowns["hook_anchor_dist"] = _dq
            return {"type": "skill", "id": 2, "mode": "tap",
                    "reason": "enemy_in_skill2_range"}
        # 一技能消耗(平时也允许, 敌英在一技能范围内)
        if not in_ult and ready("skill1_t", SKILL1_THROTTLE_S) \
                and _nq is not None and _dq <= SKILL1_RANGE_FRAC \
                and _s1q.get("unlocked") is not False and _has_yolo_eh_q:
            cooldowns["skill1_t"] = now
            cooldowns["skill"] = now
            return {"type": "skill", "id": 1, "mode": "tap", "reason": "enemy_hero_near"}
        # v2.65 鈶鏁屼汉瀛樺湪辨櫘鏀        if ready("attack_t", ATTACK_THROTTLE_S) and _has_yolo_eh_q:
            cooldowns["attack_t"] = now
            return {"type": "attack", "priority": "free", "reason": "engage_enemy"}
        # v2.71 鎶鑳介棬鑷洕(1s鑺傛祦): 鏈夋晫璇佹嵁鍗存湭鍑轰换浣曟妧鑳芥椂鎵撳嵃鍏抽敭閲        if enemies or enemy_bars:
            _sg = cooldowns.get("sg_t", 0.0)
            if now - float(_sg) > 1.0:
                cooldowns["sg_t"] = now
                print(f"[SkillGate] enemies={len(enemies)} "
                      f"reds={len(_mmr_pts)} d_red={_d_red_near:.2f} dq={_dq:.2f} "
                      f"in_ult={in_ult} s2={_s2q.get('unlocked', '')} "
                      f"s3={_s3q.get('unlocked', '')} s1={_s1q.get('unlocked', '')} "
                      f"hook_pending={round(now - float(cooldowns.get('hook_pending', 0)), 1)}")

    # ---- v2.44 鍩虹鑷喅绛栦笁: HP<20% 鎴MP<20% 涓斿畨鍏綅缃-> 寮哄埗鍥炲煄 (v2.74 钃濋噺涔熷洖瀹 ----
    if _hpq < 0.20 or _mpq < 0.20:
        _safe = not any(dist_width(s[0], s[1]) < 0.45 for s in (enemies + minions + monsters))
        _under_my_tower = any(dist_width(t[0], t[1]) < 0.35 for t in ally_turrets)
        if _safe and (_under_my_tower or not turrets) and ready("recall_t", 20.0):
            cooldowns["recall_t"] = now
            return {"type": "recall", "reason": "low_hp_safe_recall_15"}

    # ---- 0.5) 浣庤閲忔挙閫堢敤鎴疯鍒欙級欻P<20% 鍥炶嚜瀹跺涓嬪啀鍥炲煄 ----
    hp = float((state_dict.get("ui") or {}).get("hp") or 1.0)
    if hp < LOW_HP_FRAC:
        dangerous = any(dist_width(s[0], s[1]) < DANGER_FRAC
                        for s in (enemies + minions))
        camp = cooldowns.get("camp") or "blue"
        fx, fy = FOUNTAIN_DIR_BLUE if camp == "blue" else FOUNTAIN_DIR_RED
        # v2.5氭畫琛鍥炲煄瑕佸湪鑷濉斾笅堣嚜瀹跺涓績 < 0.35 灞忓夋墠瀹夊叏鍥炲煄
        nt_ally = nearest(ally_turrets)
        under_ally_tower = (nt_ally is not None
                            and dist_width(nt_ally[0], nt_ally[1]) < ALLY_TOWER_RECALL_FRAC)
        if not dangerous and under_ally_tower and ready("recall_t", RECALL_THROTTLE_S):
            cooldowns["recall_t"] = now
            return {"type": "recall", "reason": "low_hp_safe_recall"}
        # 涓嶅湪濉斾笅 -> 鏈嚜瀹跺璧帮紙鏃犲鍙鍒欐湞娉夋按> 忓湴鍥剧偣鍑 鑷娉夌偣
        return {"type": "map_move", "nx": fx, "ny": fy, "reason": "retreat_low_hp_map"}

    # ---- 0.6) 鎭閿紙鐢埛瑙勫垯夛細HP 鎴MP < 80% 涓旇韩杈瑰畨鍏鈫鎸夋仮澶嶉敭 ----
    ui = state_dict.get("ui") or {}
    mp = float(ui.get("mp") or 1.0)
    if (hp < LOW_RESOURCE_FRAC or mp < LOW_RESOURCE_FRAC) \
            and ready("restore_t", RESTORE_THROTTLE_S):
        near_danger = any(dist_width(s[0], s[1]) < DANGER_FRAC
                          for s in (enemies + minions + monsters))
        if not near_danger:
            cooldowns["restore_t"] = now
            return {"type": "restore", "reason": "low_resource_safe_restore"}

    # ---- 1) 濉旇閬匡細鏁屾柟濉斿湪杩戝鍙涓旀棤鎴戞柟忓叺(ally_minion) 鈫涓嶈繘鍏鏀诲嚮鑼冨洿 ----
    nt = nearest(turrets)
    turret_threat = bool(turrets) and not ally_minions and nt is not None \
        and dist_width(nt[0], nt[1]) < TURRET_THREAT_FRAC
    if turret_threat:
        tx, ty = nt[0], nt[1]
        # 宸插湪濉旀敾鍑昏寖鍥村唴堝涓績 < ESCAPE_FRAC夆啋 绔嬪嵆鍙嶅悜閫冪堢敤鎴峰弽棣堬細琚鎵撲笉鐭亾鍑哄幓        if dist_width(tx, ty) < TURRET_ESCAPE_FRAC:
        # inside turret fire -> escape
        if dist_width(tx, ty) < TURRET_ESCAPE_FRAC:
            cooldowns["escape_t"] = now   # v2.15 閫冭劚鎸佺画s 鍐呬繚鎸佽繙绂诲
            ax, ay = _away_map(state_dict, tx, ty)
            return {"type": "map_move", "nx": ax, "ny": ay, "reason": "escape_turret"}
        ne = nearest(enemies)
        if ne is None:
            # 鏃犳晫浜烘椂涔熶笉鏈濆鏂瑰悜璧帮紙鏈繙绂诲鏂瑰悜            ax, ay = _away_map(state_dict, tx, ty)
            return {"type": "map_move", "nx": ax, "ny": ay, "reason": "avoid_turret"}
        # 鏈夋晫浜烘椂氳嫢鏁屼汉涓庡鍚屽悜涓斿濞佽儊屼粛鍙挬堥挬瀛愪笉杩涘満夛紝绉诲姩鏂瑰悜淇濇寔浣嗘爣璁        cooldowns["turret_threat"] = 1.0
    else:
        cooldowns["turret_threat"] = 0.0
    # v2.15 閫冭劚鎸佺画歟scape 鍚1.5s 鍐呭嵆浣垮垽瀹氱煭鏆傛秷澶变篃缁画杩滅堢洿鍒扮湡姝ｅ嚭鍦堬級
    if now - float(cooldowns.get("escape_t", 0.0)) < 1.5 and turrets:
        nt2 = nearest(turrets)
        tx, ty = nt2[0], nt2[1]
        ax, ay = _away_map(state_dict, tx, ty)
        return {"type": "map_move", "nx": ax, "ny": ay, "reason": "escape_turret_hold"}

    # ---- 2) 浜屾妧鑳斤紙閽瓙夛細鏁屾柟鑻遍泟妗嗘垨鏁屼汉绾潯鍦寖鍥村唴鍗抽挬坴2.15: 绾潯=鍙鏁岃嫳闆勯搧璇侊級----
    skill_states = (state_dict.get("ui") or {}).get("skill_states") or {}
    # v2.12 MP 绠＄悊歁P < 20% 鏃剁鐢妧鑳斤紙鐪佽摑夛紝鍙敤鏅敾
    mp = float((state_dict.get("ui") or {}).get("mp") or 1.0)
    MP_SAVE_THRESHOLD = 0.20
    mp_saving = mp < MP_SAVE_THRESHOLD
    # hook signal lives only from screen enemies
    hook_signal = list(enemies)
    _mmr0 = ((state_dict.get("minimap") or {}).get("dots") or {}).get("red") or []
    if not _mmr0:
        pass
    # v2.20 开团血线
    hp = float((state_dict.get("ui") or {}).get("hp") or 1.0)
    # v2.31 hook signal
    if hook_signal and not mp_saving and hp >= SKILL_HP_MIN:
        ne = nearest(hook_signal)
        d = dist_width(ne[0], ne[1])
        if d <= SKILL2_RANGE_FRAC and ready("skill2_t", SKILL2_THROTTLE_S):
            # 鎶鑳借閿佹鏌紙v2.5夛細鏈閿侊紙鐏版殫変笉閲婃斁
            s2 = skill_states.get("2") or {}
            if s2.get("unlocked") is False:
                pass   # locked -> skip hook
            else:
#                # 閽瓙閬尅妫鏌紙鐢埛瑙勫垯 007夛細鏁屼汉杩炵嚎"绮楃洿绾涓婃湁鏁屽叺/閲庢-> 涓嶉挬
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

    # ---- 4) 绉诲姩氭寔缁嫋鍔----
    # v2.7 鐩爣婊炲洖氱洰鏍囬绻佽烦鍙橈紙钃濈偣鍣０夊鑷翠贡杞-> 鐩爣闇绋冲畾 HOLD_S 绉掓墠鍒囨崲
    TARGET_HOLD_S = 1.5
    if enemies and hp < CHASE_BREAK_FRAC:
        camp = cooldowns.get("camp") or "blue"
        fx, fy = FOUNTAIN_DIR_BLUE if camp == "blue" else FOUNTAIN_DIR_RED
        target = (fx, fy)
        reason = "stop_chase_low_hp_retreat"
        if now - float(cooldowns.get("skill2_t", 0.0)) < HOOK_FLIGHT_S:
            return {"type": "none", "reason": "hook_flight"}
        return {"type": "map_move", "nx": fx, "ny": fy, "reason": reason}
    target = None
    reason = None
    if enemies:
        ne = nearest(enemies)
        # v2.21 鎺垬寮曟搸氭晫浜哄湪 0.35 灞忓鍐-> 浼樺厛璐磋韩缂犳枟(绉诲姩+鏀诲嚮閿繛鍙
        d_ne = dist_width(ne[0], ne[1])
        if d_ne < ENGAGE_FRAC and ready("attack_t", ATTACK_THROTTLE_S):
            cooldowns["attack_t"] = now
            return {"type": "attack", "priority": "free", "reason": "engage_enemy"}
        target = (ne[0], ne[1])
        reason = "chase_enemy"
    else:
        # enemy bar chase (hysteresis)
        mm0 = state_dict.get("minimap") or {}
        mm_red0 = (mm0.get("dots") or {}).get("red") or [] if mm0.get("found") else []
        bar_hold = now - float(cooldowns.get("bar_chase_t", 0.0)) < 1.5
        if enemy_bars and mm_red0:
            nearest_bar = min(enemy_bars, key=lambda b: math.hypot(
                b[0] / w - 0.5, (b[1] / h - 0.5) * (h / w)))
            bx, by = nearest_bar[0] / w, nearest_bar[1] / h
            if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0:
                target = (bx, by)
                reason = "chase_enemy_bar"
                cooldowns["bar_chase_t"] = now
                cooldowns["bar_target"] = (bx, by)
        elif bar_hold and cooldowns.get("bar_target") and mm_red0:
            target = tuple(cooldowns["bar_target"])
            reason = "chase_enemy_bar"
        if target is None:
            # movement decision
            camp = cooldowns.get("camp") or "blue"
            lane = LANE_DIR_BLUE if camp == "blue" else LANE_DIR_RED
            mm = state_dict.get("minimap") or {}
            mm_blue, mm_red = [], []
            if mm.get("found"):
                mm_blue = (mm.get("dots") or {}).get("blue") or []
                mm_red = (mm.get("dots") or {}).get("red") or []

            # opening safe: first 6s follow lane
            match_t = float(cooldowns.get("match_start_t", 0.0))
            in_opening = match_t > 0 and now - match_t < 6.0
            # v2.33 鎬佸娍闂帶(瀛敤鎴: 鍧囧娍鏁墦(0.55)/鍔ｅ娍榫熺缉(0.65)/浼樺娍鍘嬭蛋(0.60); 寮灞鏇寸
            _u_eng_hp = 0.55 if len(mm_blue) == len(mm_red) else (
                0.65 if len(mm_red) > len(mm_blue) else 0.60)
            if in_opening:
                _u_eng_hp = max(_u_eng_hp, 0.70)
            if len(mm_red) >= 6:      # enemy herd: no support
                _u_eng_hp = 1.01
            # revive 5s: follow only
            if now - float(cooldowns.get("revive_t", 0.0)) < 5.0 and cooldowns.get("revive_t"):
                _u_eng_hp = 1.01
            # v2.39 杈呭姪璺嚎(鐢埛: 涓笅璺悇45%): 涓嬭矾璧板粖(钃濇柟鍙充笅/绾柟宸笂) 浼樺厛,
            # (comment)
            _mid_c = (0.5, 0.58) if camp == "blue" else (0.5, 0.42)
            def _corridor(p, c, k=0.30):
                return math.hypot(p[0] - c[0], p[1] - c[1]) <= k
            def _pick_lane(down_only=None):
                return None
            _camp0 = cooldowns.get("camp") or "blue"
            _down_c = (0.75, 0.72) if _camp0 == "blue" else (0.25, 0.28)
            _mid_c = (0.5, 0.58) if _camp0 == "blue" else (0.5, 0.42)
            na = nearest(allies)
            # v3.2 支援=跟随队友(用户铁律): 小地图红点不可作为支援目标(检测不准);
            #      队友蓝点在身边(走廊内/任意)就去队友处; 只有完全无蓝点才考虑走廊红点支援
            if not in_opening and mm_blue:
                self_g = (mm.get("dots") or {}).get("green") or []
                myx, myy = self_g[0] if self_g else (0.5, 0.5)
                _down_blue = [p for p in mm_blue if _corridor(p, _down_c, 0.35)]
                _mid_blue = [p for p in mm_blue if _corridor(p, _mid_c, 0.35)]
                _pool_b = _down_blue or _mid_blue or mm_blue
                _trb = min(_pool_b, key=lambda p: (p[0]-myx) ** 2 + (p[1]-myy) ** 2)
                target = (_trb[0], _trb[1])
                reason = "support_ally_engaged" if (_corridor(_trb, _down_c, 0.35)
                                                    or _corridor(_trb, _mid_c, 0.35)) else "follow_ally_minimap"
            elif not in_opening and mm_red and hp >= _u_eng_hp:
                # 无任何队友蓝点 -> 走廊红点支援(发育路/中路), 河道红点不去
                self_g = (mm.get("dots") or {}).get("green") or []
                myx, myy = self_g[0] if self_g else (0.5, 0.5)
                _down_reds = [p for p in mm_red if _corridor(p, _down_c)]
                _mid_reds = [p for p in mm_red if _corridor(p, _mid_c)]
                _pool = _down_reds or _mid_reds
                tr = min(_pool, key=lambda p: (p[0] - myx) ** 2 + (p[1] - myy) ** 2) \
                    if _pool else None
                d_red = math.hypot(tr[0] - myx, tr[1] - myy) if tr else 9.9
                if tr is not None and d_red < 0.60:
                    target = (tr[0], tr[1])
                    reason = "support_red_nearest"
                else:
                    target = None
            if target is None and not in_opening and mm_red and hp < _u_eng_hp:
                # low hp: follow ally
                na_tmp = nearest(allies)
                if na_tmp is not None:
                    target = (na_tmp[0], na_tmp[1])
                    reason = "follow_ally_lowhp"
            # v3.1 用户铁律: 不蹲草! 支援=去队友身边, 红点近也在移动中->交给技能块/移动
            if target is None and na is not None and not in_opening:
                # 璺熷皠鎵闃熷弸氬睆骞ally_hero 鐩存帴璺熼殢涜创杩戝埌 0.12 灞忓鎵嶅仠(鐢埛鍑犱箮涓嶅仠)
                d_ally = dist_width(na[0], na[1])
                if d_ally < 0.12 and not mm_red:
                    return {"type": "none", "reason": "follow_ally_hold"}
                # v3.0 支援=跟随队友(用户): 屏内队友无条件跟(三条路都支援), 只避河道中央站桩
                target = (na[0], na[1])
                reason = "follow_ally"
            elif mm_blue and not in_opening:
                # 璺熼槦鐩爣: 涓嬭矾璧板粖钃濈偣浼樺厛(勬墜), 娆￠変腑璺 璐績(闃插崟鐐逛吉妫)
                _down_blue = [p for p in mm_blue if _corridor(p, _down_c, 0.35)]
                _mid_blue = [p for p in mm_blue if _corridor(p, _mid_c, 0.35)]
                # v3.0 支援=跟随队友: 发育路/中路走廊蓝点优先, 无则跟最近蓝点(三条路都支援);
                #      河道中央带(r<0.10 距0.5,0.5)蓝点不跟(队友在河道=不陪他站桩, 去发育路)
                _pool_b = _down_blue or _mid_blue
                if not _pool_b:
                    _sg3 = (mm.get("dots") or {}).get("green") or []
                    _px3, _py3 = _sg3[0] if _sg3 else (0.5, 0.5)
                    _nb3 = min(mm_blue, key=lambda p: (p[0]-_px3)**2 + (p[1]-_py3)**2)
                    if math.hypot(_nb3[0]-0.5, _nb3[1]-0.5) < 0.10:
                        lx, ly = lane
                        target = (lx, ly)
                        reason = "lane_develop_hold"
                    else:
                        target = (_nb3[0], _nb3[1])
                        reason = "follow_ally_minimap"
                else:
                    lx = sum(p[0] for p in _pool_b) / len(_pool_b)
                    ly = sum(p[1] for p in _pool_b) / len(_pool_b)
                    target = (lx, ly)
                    reason = "follow_ally_minimap"
            else:
                lx, ly = lane
                target = (lx, ly)
                reason = "lane_develop"
    # v3.2 用户兜底铁律: 决策不明时跟随队友去清兵线(不空转/不河道游荡)
    if reason == "lane_develop" or reason == "lane_develop_hold":
        if ally_minions:
            _am = min(ally_minions, key=lambda s: dist_width(s[0], s[1]))
            target = (_am[0], _am[1])
            reason = "follow_ally_clearlane"
        elif mm_blue:
            _sgf = (mm.get("dots") or {}).get("green") or []
            _fx, _fy = _sgf[0] if _sgf else (0.5, 0.5)
            _nbf = min(mm_blue, key=lambda p: (p[0]-_fx)**2 + (p[1]-_fy)**2)
            target = (_nbf[0], _nbf[1])
            reason = "follow_ally_minimap"
    if target is None:
        return {"type": "none", "reason": "no_target"}
    # v2.7 鐩爣婊炲洖氳摑鐐瑰櫔澹板鑷寸洰鏍囨瘡甯烦鍙-> 鐩爣闇绋冲畾 HOLD_S 绉掓墠鍒囨崲
    # 堜粎瀵瑰皬鍦板浘绫荤洰鏍囩敓鏁堬紱chase_enemy 杩芥晫浜轰笉婊炲洖岄伩鍏嶈拷涓級
    if reason in ("follow_ally_minimap", "support_red_centroid", "support_red_nearest",
                  "support_ally_engaged", "lane_develop", "follow_ally_lowhp"):
        prev = cooldowns.get("nav_target")
        prev_t = float(cooldowns.get("nav_target_t", 0.0))
        if prev is not None:
            dist = math.hypot(target[0] - prev[0], target[1] - prev[1])
            if dist > 0.12 and now - prev_t < TARGET_HOLD_S:
                # 鐩爣璺冲彉浣嗘湭鍒板垏鎹湡 -> 缁存寔鏃洰鏍囷紙鍒锋柊鏃堕棿鎴筹紝閬垮厤姘镐箙 hold                target = tuple(prev)
                reason = f"{reason}_hold"
                cooldowns["nav_target"] = tuple(target)
                cooldowns["nav_target_t"] = now
            else:
                cooldowns["nav_target"] = tuple(target)
                cooldowns["nav_target_t"] = now
        else:
            cooldowns["nav_target"] = tuple(target)
            cooldowns["nav_target_t"] = now
    # 閽瓙椋炶鏈熷仠姝鍔紙绛夊嬀涓粨鏋滐紝闃茶嚜宸辫蛋杩戣鍒繛鎷涳級
    if now - float(cooldowns.get("skill2_t", 0.0)) < HOOK_FLIGHT_S:
        return {"type": "none", "reason": "hook_flight"}
    # v2.27 忓湴鍥剧偣鍑荤鍔鐜嬭呰緟鍔鍔: 鍦板浘绫荤洰鏍囩洿鎺tap 忓湴鍥剧偣, 寮曟搸鑷姩瀵昏矾
    if reason in ("follow_ally_minimap", "support_red_centroid", "support_red_nearest",
                  "support_ally_engaged", "lane_develop", "follow_ally_lowhp",
                  "support_red_centroid_hold", "follow_ally_minimap_hold",
                  "support_red_nearest_hold", "support_ally_engaged_hold",
                  "lane_develop_path", "follow_ally_lowhp_hold",
                  "follow_ally_minimap_path", "support_red_nearest_path",
                  "support_ally_engaged_path", "lane_develop_hold",
                  "follow_ally_clearlane", "follow_ally"):
        return {"type": "map_move", "nx": max(0.02, min(0.98, target[0])),
                "ny": max(0.02, min(0.98, target[1])), "reason": reason}
    # dual-track movement
    STICK_REASONS = ("chase_enemy", "chase_enemy_bar", "follow_ally",
                     "escape_turret", "escape_turret_hold")
    mm = state_dict.get("minimap") or {}
    if reason in STICK_REASONS:
        theta = math.atan2(-(target[1] - 0.5) * aspect, target[0] - 0.5)
        return {"type": "move", "theta": theta, "r": MOVE_R,
                "duration_ms": MOVE_DURATION_MS, "reason": reason}
    # 鍏朵綑=忓湴鍥剧偣鍑 灞忓箷绫荤洰鏍囨槧勫埌鏈杩戠孩/钃濈偣; 閫閬垮鏄犲皠鍙嶅悜鐐 鎾=娉夌偣
    mmp_dots = (mm.get("dots") or {}) if mm.get("found") else {}
    greens = mmp_dots.get("green") or []
    blues = mmp_dots.get("blue") or []
    reds_ = mmp_dots.get("red") or []
    gx, gy = greens[0] if greens else (0.5, 0.5)
    nx, ny = target[0], target[1]
    if reason.startswith(("follow_ally_minimap", "support", "lane", "follow_ally_lowhp")):
        pass  # 宸叉槸鍦板浘鍧愭爣
    elif reason.startswith(("retreat",)):
        fx, fy = FOUNTAIN_DIR_BLUE if (cooldowns.get("camp") or "blue") == "blue" \
            else FOUNTAIN_DIR_RED
        nx, ny = fx, fy
    elif reason.startswith(("escape_turret", "avoid_turret")):
        tx, ty = nt[0], nt[1]
        ax, ay = _away_map(state_dict, tx, ty)
        nx, ny = ax, ay
    return {"type": "map_move", "nx": max(0.02, min(0.98, nx)),
            "ny": max(0.02, min(0.98, ny)), "reason": reason}


# ---------------------------------------------------------------------------
# 鎵杈呭姪堜粎涓诲惊鐜娇鐢紝淇濇寔 decide 绾嚱鏁帮級
# ---------------------------------------------------------------------------

def update_cooldowns(action: dict, cooldowns: dict, now: float):
    """decoded docstring."""
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


def _async_result_detect(frame, reward):
    """decoded docstring."""
    import threading as _th

    def worker(fr):
        try:
            import subprocess as _sp
            import sys as _sys
            import cv2 as _cv2
            tmp = ROOT / "temp" / "result_live.png"
            _cv2.imwrite(str(tmp), fr)
            prompt = ("This is the WZRY match result screen. "
                      "If victory (gold text) answer victory. "
                      "If defeat answer defeat. "
                      "Otherwise answer unknown. "
                      "Output one word only.")
            r = _sp.run(
                [_sys.executable, "-X", "utf8",
                 str(ROOT / "scripts" / "train" / "modlens_ask.py"),
                 str(tmp), prompt],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=90)
            txt = (r.stdout or "").strip().lower()
            if "victory" in txt or "victory" in txt:
                sc = reward.on_event("victory")
                print(f"[{datetime.now():%H:%M:%S}] [反馈] 事件 victory {sc:+.0f} 分 "
                      f"(总分 {reward.total:.0f})")
            elif "defeat" in txt:
                sc = reward.on_event("defeat")
                print(f"[{datetime.now():%H:%M:%S}] [反馈] 事件 defeat {sc:+.0f} 分 "
                      f"(总分 {reward.total:.0f})")
        except Exception:
            pass

    _th.Thread(target=worker, args=(frame.copy(),), daemon=True).start()


def _rec_hook_measure(cooldowns: dict, state_dict: dict, action: dict, hit: bool):
    """decoded docstring."""
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
    """decoded docstring."""
    now = time.time()
    t = action.get("type")
    if t == "map_move":
        # v2.27 鐜嬭鐐瑰嚮忓湴鍥捐嚜鍔綅绉杈呭姪绉诲姩: 鐩存帴鐐瑰皬鍦板浘鐩爣
        px = int(float(action.get("nx", 0.5)) * 242)
        py = int(float(action.get("ny", 0.5)) * 242)
        ex.tap(px, py, source="policy")
    elif t == "move":
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
                time.sleep(0.12)   # combo interval
    update_cooldowns(action, cooldowns, now)


# ---------------------------------------------------------------------------
# 瀹炴椂鍒嗘瀽鏍囨敞坴2.15: 姣忓抚鍏绱犳爣娉-> temp/live_annot/屼緵浜哄伐鏍告煡 AI 鐞嗚# ---------------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# recovered consts (from v3.1)
RECALL_THROTTLE_S = 20.0
RECALL_ACTIVE_S = 8.0
TURRET_SAFE_FRAC = 0.55
TURRET_THREAT_FRAC = 0.45
ENGAGE_FRAC = 0.30

_FONT_CACHE = {}


def _lfont(size):
    if size not in _FONT_CACHE:
        for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
                   r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
            if Path(fp).exists():
                _FONT_CACHE[size] = ImageFont.truetype(fp, size)
                break
        else:
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]


def _ltext(img, text, org, size=16, color=(255, 255, 0)):
    f = _lfont(size)
    if f is None:
        return
    x, y = int(org[0]), int(org[1])
    h, w = img.shape[:2]
    if x >= w or y >= h:
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    dr.text((x, y), text, font=f, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


_LA_DIR = None
_LA_LAST = 0.0
MM_LABELS_CN = {"self": "self", "ally": "ally", "enemy": "enemy",
                "monster": "monster", "buff": "buff", "minion": "minion"}
CLS_CN = {"enemy_hero": "enemy", "ally_hero": "ally", "enemy_minion": "enemy_minion",
          "ally_minion": "ally_minion", "enemy_turret": "enemy_turret", "ally_turret": "ally_turret",
          "enemy_crystal": "enemy_crystal", "ally_crystal": "ally_crystal",
          "neutral_monster": "monster", "hook_aim": "hook", "skill_effect": "skill",
          "self": "self"}


def render_live_analysis(frame, dets, mm, state_dict, action, beh, score, now):
    """decoded docstring."""
    global _LA_DIR, _LA_LAST
    if _LA_DIR is None:
        _LA_DIR = ROOT / "temp" / "live_annot"
        _LA_DIR.mkdir(parents=True, exist_ok=True)
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 鍏睆 12 绫绘 (v2.56: self 妗嗗交搴曚笉鐢烩斺旂敤鎴峰凡纭鑷繁鍦睆骞曚腑澶
    for d in dets:
        if d.cls == "self":
            continue
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        col = {"enemy_hero": (0, 0, 255), "ally_hero": (255, 160, 0),
               "enemy_minion": (0, 100, 255), "ally_minion": (120, 220, 0),
               "enemy_turret": (0, 0, 200), "ally_turret": (220, 140, 0),
               "enemy_crystal": (255, 0, 200), "ally_crystal": (200, 255, 0),
               "neutral_monster": (0, 255, 255), "hook_aim": (255, 0, 255),
               "skill_effect": (255, 255, 0)}.get(d.cls, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        _ltext(vis, f"{CLS_CN.get(d.cls, d.cls)} {d.conf:.2f}",
               (x1 + 2, max(2, y1 - 20) if y1 > 22 else y2 + 2), 14, col)

    # 鑷繁 HP/MP
    ui = state_dict.get("ui") or {}
    hp = float(ui.get("hp") or 0.0)
    mp = float(ui.get("mp") or 0.0)
    # v2.80 hero 全屏标记已删: 自己在屏幕中央固定, 不需要检测/显示 (仅下方血量数字)
    _dummy_hero = None

    # 忓湴鍥惧叏瑕佺礌 2x堝彸涓婏級v2.25: 鍏箙 0-242 (鍚笅杈规娉夋按鍖
    v7 = (mm or {}).get("v7") or {}
    if v7:
        bx, by, bsz = w - 490, 34, 480
        cv2.rectangle(vis, (bx, by), (bx + bsz, by + bsz), (40, 40, 40), -1)
        mmcrop = vis[by + 2:by + bsz - 2, bx + 2:bx + bsz - 2]
        mmcrop[:] = cv2.resize(frame[0:242, 0:242], (bsz - 4, bsz - 4))
        scale = (bsz - 4) / 242.0

        def dpos(n):
            return (int(bx + 2 + n[0] * 242 * scale), int(by + 2 + n[1] * 242 * scale))
        for d in v7.get("self", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 12, (0, 255, 0), 3)
            _ltext(vis, "self", (px + 13, py - 8), 13, (0, 255, 0))
        for d in v7.get("ally", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 10, (255, 150, 0), -1)
            _ltext(vis, "ally", (px + 11, py - 8), 13, (255, 150, 0))
        for d in v7.get("enemy", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 10, (0, 0, 255), -1)
            _ltext(vis, "enemy", (px + 11, py - 8), 13, (255, 80, 80))
        for d in v7.get("monster", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 7, (0, 255, 255), -1)
        for d in v7.get("buff", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 7, (255, 0, 255), -1)
        tws = (mm or {}).get("towers_v7") or {}
        for d in tws.get("ally", []):
            px, py = dpos(d["n"]); cv2.rectangle(vis, (px - 8, py - 8), (px + 8, py + 8),
                                                 (255, 180, 0), -1)
        for d in tws.get("enemy", []):
            px, py = dpos(d["n"]); cv2.rectangle(vis, (px - 8, py - 8), (px + 8, py + 8),
                                                 (0, 80, 255), -1)
        for d in (v7.get("minions") or {}).get("ally", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 4, (150, 255, 150), -1)
        for d in (v7.get("minions") or {}).get("enemy", []):
            px, py = dpos(d["n"]); cv2.circle(vis, (px, py), 4, (150, 150, 255), -1)

    # 搴曢儴淇伅鏉    beh_label = (beh or {}).get("label", "")
    act_txt = f"鍐崇瓥: {format_decision(action)}"
    reason = action.get("reason", "")
    mmc = (mm or {}).get("dots") or {}
    info = (f"beh: {beh.get('label', '')} | {reason} | score {score:.0f} | "
            f"map blue {len(mmc.get('blue', []))}/red {len(mmc.get('red', []))}/"
            f"green {len(mmc.get('green', []))} | {datetime.now():%H:%M:%S}")
    cv2.rectangle(vis, (0, h - 64), (w, h), (0, 0, 0), -1)
    _ltext(vis, info, (10, h - 58), 17, (0, 255, 255), )

    # 钀界洏
    cv2.imwrite(str(_LA_DIR / "annot_latest.png"), vis)
    if now - _LA_LAST >= 1.0:
        _LA_LAST = now
        cv2.imwrite(str(_LA_DIR / f"annot_{now:.0f}.png"), vis)
        # 娓呯悊鏃枃浠淇濈暀200)
        old = sorted(_LA_DIR.glob("annot_*.png"))
        for f in old[:-210]:
            try:
                f.unlink()
            except OSError:
                pass
    # 琛屼负瀹
    try:
        rec = {"t": now, "ts": datetime.now().isoformat(timespec="seconds"),
               "label": beh_label, "action": action.get("type"),
               "id": action.get("id"), "reason": reason,
               "hp": round(hp, 3), "mp": round(mp, 3), "score": round(float(score), 1),
               "mm": {"blue": len(mmc.get("blue", [])), "red": len(mmc.get("red", [])),
                      "green": len(mmc.get("green", []))}}
        with open(_LA_DIR / "behavior.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return vis


def format_decision(action: dict) -> str:
    t = action.get("type")
    reason = action.get("reason", "")
    if t == "none":
        return f"none [{reason}]"
    if t == "skill":
        return f"skill{action.get('id')} ({action.get('mode', 'tap')}) [{reason}]"
    if t == "summoner":
        return f"summoner [{reason}]"
    if t == "recall":
        return f"recall [{reason}]"
    if t == "restore":
        return f"restore [{reason}]"
    if t == "combo":
        names = [f"{a.get('type')}{a.get('id', '')}" for a in action.get("actions", [])]
        return f"combo {'+'.join(names)} [{reason}]"
    if t == "move":
        return f"move theta={action.get('theta', 0.0):+.2f} r={action.get('r', 0.0)} " \
               f"{action.get('duration_ms', 0)}ms [{reason}]"
    return f"{t} [{reason}]"


def main():
    ap = argparse.ArgumentParser(description="doc")
    ap.add_argument("--capture", default="desktop", choices=["desktop", "screencap", "scrcpy"],
                    help="frame source: desktop/screencap/scrcpy")
    ap.add_argument("--seconds", type=float, default=120.0, help="run seconds")
    ap.add_argument("--forever", action="store_true",
                    help="opt")
    ap.add_argument("--model", default=str(ROOT / "runs" / "detect" / "zhongkui_11cls"
                                           / "weights" / "best.pt"),
                    help="opt")
    ap.add_argument("--action", action="store_true",
                    help="opt")
    ap.add_argument("--no-action", dest="no_action", action="store_true",
                    help="opt")
    ap.add_argument("--save", action="store_true", help="opt")
    ap.add_argument("--detect-hz", type=float, default=10.0, help="opt")
    ap.add_argument("--show", action="store_true", help="opt")
    args = ap.parse_args()

    do_action = bool(args.action) and not args.no_action

    from wzry.calib import load_calibration
    from wzry.capture.scrcpy_stream import ScrcpyStreamCapture
    from wzry.data.collector import MatchRecorder
    from wzry.state.fuser import build_state
    from wzry.state.match_state import MatchPhase, MatchStateMachine
    from wzry.vision.detector import YoloDetector
    from wzry.vision.mm_tracker_v7 import MMTrackerV7

#    # v2.8歳otate_s 璋冨鍒30 鍒嗛挓堥伩鍏嶅灞涓疆杞鑷鏃犲抚"鍋滈】    # v2.47 閲囬泦: desktop=MuMu绐楀彛PrintWindow骞虫粦娴榛樿, ~12fps鏃犲垎娈
    if args.capture == "desktop":
        from wzry.capture.desktop_capture import DesktopCapturePrint
        cap = DesktopCapturePrint(interval=0.02)
        print("启动 MuMu 窗口平滑视频流(PrintWindow, ~84ms/帧)...")
    elif args.capture == "screencap":
        from wzry.capture.screencap_capture import ScreencapCapture
        cap = ScreencapCapture(interval=0.02, rotate_s=1800)
        print("启动 screencap 实时采集（adb 直读）...")
    else:
        from wzry.capture.scrcpy_stream import ScrcpyStreamCapture
        cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy", rotate_s=1800)
        print("启动 scrcpy 流...")
    cap.start()

    calib, _ = load_calibration()
    sm = MatchStateMachine(minimap_center_norm=calib.get("minimap_center", [0.086, 0.129]))

    print(f"鍔犺浇妫娴嬫鍨{args.model} ...")
    det = YoloDetector(args.model, conf=0.35)

    # 忓湴鍥捐窡韪細鍏堥獙 = 鏍噯鐐癸紙褰掍竴鍖-> 棣栦釜瀹為檯甯昂瀵告崲绠楋級
    mm_prior = None

    def make_tracker(frame):
        nonlocal mm_prior
        if mm_prior is None:
            h, w = frame.shape[:2]
            mc = calib.get("minimap_center", [0.086, 0.129])
            mm_prior = [int(mc[0] * w), int(mc[1] * h)]
        from wzry.vision.mm_tracker_v7 import MMTrackerV7
        return MMTrackerV7()

    tracker = None
    from wzry.vision.wall_sensor import WallSensor
    wall_sensor = WallSensor()

    if do_action:
        from wzry.action.executor_v2 import ActionExecutor
        ex = ActionExecutor()
        print("鍔綔妯紡: 鎵-action岀湡鏈鸿鎽革級")
    else:
        ex = None
        print("鍔綔妯紡: 浠呰瀵燂紙榛樿 --no-action屼笉娉叆浠讳綍瑙懜涘姞 --action 鎵嶆墽琛岋級")

    recorder = MatchRecorder(base_dir=ROOT / "data" / "matches")
    from wzry.understanding.behavior import BehaviorTagger
    from wzry.feedback.reward import RewardSystem
    tagger = BehaviorTagger()
    reward = RewardSystem()
    cooldowns = {"skill1_t": 0.0, "skill2_t": 0.0, "skill3_t": 0.0,
                 "summoner_t": 0.0, "recall_t": 0.0, "hp_t": 0.0,
                 "restore_t": 0.0, "attack_t": 0.0,
                 "skill": 0.0, "hook_pending": 0.0,
                 "hook_anchor_dist": 0.0, "turret_threat": 0.0,
                 "hook_blocked": 0.0, "match_start_t": 0.0,
                 "roster": None, "roster_pending": 0.0}
    # 鎶鑳芥寜閽儚绱犲潗鏍囷紙read_ui 鐢級
    with open(ROOT / "configs" / "calibration_absolute.json", encoding="utf-8") as _f:
        _pts = json.load(_f)["points"]
    skills_pts = {1: _pts["skill1"], 2: _pts["skill2"], 3: _pts["skill3"]}

    # confirm frames
    CONFIRM_FRAMES = 2
    confirm_streak = 0
    confirmed = False

    detect_interval = 1.0 / max(0.5, args.detect_hz)
    last_detect = 0.0
    last_log = 0.0
    last_sig = ("none", None)
    frame_id = 0
    prev_phase = None
    n_frames = 0
    # v2.81 自我诊断(截图留存, 供 DeepSeek 视觉审检)
    try:
        from wzry.diagnose.self_check import SelfCheckDiagnostician
        self_check = SelfCheckDiagnostician(interval_s=20.0)
        self_check.start()
    except Exception:
        self_check = None
    # v2.83 行为习惯学习器(对面/队友/自己)
    try:
        from wzry.learning.behavior_mine import BehaviorMiner
        behavior_miner = BehaviorMiner()
        behavior_miner.start()
    except Exception:
        behavior_miner = None
    n_ticks = 0
    n_actions = 0
    infer_sum = 0.0
    _last_move = [None, None, 0.0]
    _hero_cls_mem = {}
    last_dead = False
    try:
        from wzry.learning.online_learner import OnlineLearner
        learner = OnlineLearner()
    except Exception:
        learner = None
    # yolo every 3 ticks
    yolo_every = 3
    yolo_count = 0
    cached_dets = []
    t_end = time.time() + args.seconds if not args.forever else float("inf")

    print("M2 Agent v2 杩愯涓紙Ctrl+C 閫鍑猴級...\n")
    try:
        while time.time() < t_end:
            frame, lag_ms = cap.wait_frame(timeout=2.0)
            if frame is None:
##                print("  鏃犲抚堟祦寮傚父)
                continue
            n_frames += 1
            phase = sm.update(frame)
            now = time.time()
            # v2.81 self-diagnose feed (AI 审检截图留存)
            try:
                if self_check is not None:
                    self_check.feed(frame, dets)
            except Exception:
                pass
            if phase != MatchPhase.IN_MATCH:
                # v3.4 用户铁律: 选人界面自己在上排=蓝方/下排=红方(等进游戏前先记阵营)
                #   检测: 上排名牌带(y90-165) 与 下排名牌带(y480-560) 白色高亮文字像素,
                #   哪排有"Starliit-001"白色名牌(另一排是敌方黑名牌) -> 自己在那排
                if not cooldowns.get("camp_row_stale", True) is False:
                    pass
                _sel_row = None
                try:
                    # 用户铁律: "Starliit-001 名牌在上排=蓝方 / 下排=红方"
                    # 名牌检测: 白色文字行投影 (纯白 V>200 且低饱和), 行峰即名字带
                    _hsvS = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    _H_S_ = _hsvS[..., 0].astype(int)
                    _S_S_, _V_S_ = _hsvS[..., 1].astype(int), _hsvS[..., 2].astype(int)
                    _wh_s = ((_S_S_ < 60) & (_V_S_ > 200)).astype(np.uint8)
                    # 行投影(排除顶部栏/底栏, x 150-1130)
                    _rowsum = _wh_s[:, 150:1130].sum(axis=1)
                    # 找 3 个最强连续行带(名字行 y): 分值=该行像素
                    _top_rows = sorted(range(720), key=lambda y: -int(_rowsum[y]))[:40]
                    # 名字带需两侧有字符分布(玩家名宽 60-160px), 取带峰值>80px 的行
                    _cand_rows = [y for y in _top_rows if int(_rowsum[y]) >= 80]
                    if _cand_rows:
                        _name_y = int(np.median(_cand_rows))
                        _sel_row = "blue" if _name_y < 360 else "red"
                except Exception:
                    _sel_row = None
                # v3.6 门控: 3帧稳定名牌行才锁(防结算/大厅白色文字误判), 且仅 WAITING 阶段
                if _sel_row:
                    _prev_row = cooldowns.get("sel_row_prev")
                    _row_streak = int(cooldowns.get("sel_row_streak", 0))
                    if _prev_row == _sel_row:
                        _row_streak += 1
                    else:
                        _row_streak = 1
                    cooldowns["sel_row_prev"] = _sel_row
                    cooldowns["sel_row_streak"] = _row_streak
                    if _row_streak >= 3 and not cooldowns.get("camp"):
                        cooldowns["camp"] = _sel_row
                        cooldowns.pop("camp_votes", None)
                        print(f"[{datetime.now():%H:%M:%S}] 阵营判断(选人排位3帧): "
                              f"{'蓝方' if _sel_row == 'blue' else '红方'} → "
                              f"发育路方向 {LANE_DIR_BLUE if _sel_row == 'blue' else LANE_DIR_RED}")
                # post match detect
                if prev_phase == MatchPhase.IN_MATCH and phase == MatchPhase.POST_MATCH:
                    _async_result_detect(frame, reward)
                prev_phase = phase
                confirm_streak = 0
                confirmed = False
                _last_move[:] = (None, None, 0.0)   # 闈炲灞鏉惧紑鎽囨潌
                if recorder.active:
                    recorder.close()
                    print(f"[{datetime.now():%H:%M:%S}] 瀵瑰眬缁撴潫屼細璇濆凡褰掓。")
                # learning log dump (throttled: cooldown 60s + learner 60s)
                try:
                    if learner is not None and \
                            now - float(cooldowns.get("post_dump_t", 0.0)) > 60.0:
                        cooldowns["post_dump_t"] = now
                        _dk = learner.dump_log(str(ROOT / "docs" / "LEARNING_LOG.md"))
                        if _dk:
                            print(f"[{datetime.now():%H:%M:%S}] [学习] 本局学习日志 -> "
                                  f"docs/LEARNING_LOG.md")
                except Exception:
                    pass
                continue
            # v2.26 杩炵画鎽囨潌: 鍐崇瓥闂撮殭涓嶈鎽囨潌鏉炬墜(琛紳 200ms)
            if do_action and _last_move[0] is not None and time.time() < _last_move[2]:
                try:
                    ex.move(_last_move[0], _last_move[1], 200)
                except Exception:
                    pass
            if now - last_detect < detect_interval:
                continue
            last_detect = now

            # ---- 纭鍒----
            if tracker is None:
                tracker = make_tracker(frame)
            mm = tracker.update(frame)
            if not mm["found"]:
                if not confirmed:
                    confirm_streak = 0
                    print(f"[{datetime.now():%H:%M:%S}] 対局确认中（小地图暂未定位）...")
                continue
            confirm_streak += 1
            if confirm_streak < CONFIRM_FRAMES:
                confirmed = False
                print(f"[{datetime.now():%H:%M:%S}] 瀵瑰眬纭涓{confirm_streak}/{CONFIRM_FRAMES} ...")
                continue
            # v3.0 缁挎潯閾佽瘉氶娆＄璁姹傛弧琛缁挎潯0.5 = 鏉>57px夆斺            # 瀵瑰眬鍦硥姘村嚭鐢熷繀婊涗富椤缁撶畻璇缁挎潯浠26-29px 琚帓闄紱
            # 宸茬璁悗涓嶅啀妫鏌紙闃典骸/娈嬭鏃舵棤婊鏉絾浠嶅湪瀵瑰眬涓級
            # v2.51 蹇熼氶亾: 忓湴鍥惧凡妫鍑哄崟浣钃绾缁跨偣) => 瀵瑰眬閾佽瘉, 鐩存帴纭(涓嶇瓑缁挎潯)
            if not confirmed:
                from wzry.vision.self_bars import self_hp_mp
                _hp, _mp, _pos = self_hp_mp(frame)
                _fast = False
                try:
                    _dots = (mm.get("dots") or {})
                    _fast = bool(_dots.get("blue") or _dots.get("red") or
                                 _dots.get("green")) and mm.get("found")
                except Exception:
                    _fast = False
                if _fast:
                    print(f"[{datetime.now():%H:%M:%S}] 瀵瑰眬纭(忓湴鍥鹃搧璇")
                    confirmed = True
                    confirm_streak = CONFIRM_FRAMES
                elif _hp is None or _hp < 0.5:
                    confirm_streak = 0
                    print(f"[{datetime.now():%H:%M:%S}] 瀵瑰眬纭涓紙鏃犳弧琛缁挎潯岄潪瀵瑰眬鐢婚潰..")
                    continue
            confirmed = True
            # match start time
            if not cooldowns.get("match_start_t"):
                cooldowns["match_start_t"] = now
                # v2.95 新对局开始: 清零行为层死亡残留(上局 _dead_since 污染 -> 开局假died)
                try:
                    tagger.reset_match()
                except Exception:
                    pass

            # ---- 寮灞闃佃惀鍒柇 v3.2: 多帧投票制(单帧易误判->红方被送对抗路) ----
            # 票源: ① 小地图自已绿点基地角落(右下=蓝 左上=红) 权重2 ② 小地图角落水晶色 权重1
            #   ③ 屏幕中心泉水ROI 权重1; 连续投票多数>=3 才锁; 锁前一律走发育路中立方向等待
            if not cooldowns.get("camp"):
                _votes = cooldowns.setdefault("camp_votes", {"blue": 0, "red": 0})
                _vote_t = float(cooldowns.get("camp_vote_t", 0.0))
                if now - _vote_t > 0.4:   # 0.4s 采样间隔
                    cooldowns["camp_vote_t"] = now
                    _cand = None
                    greens = (mm.get("dots") or {}).get("green") or []
                    if greens:
                        gx, gy = greens[0]
                        # 蓝方基地小地图右下 / 红方基地左上
                        _cand = "blue" if (gx > 0.35 and gy > 0.35) else "red"
                        _votes[_cand] = _votes.get(_cand, 0) + 2
                    try:
                        _mm0 = frame[0:232, 0:232]
                        _br = _mm0[176:232, 176:232].reshape(-1, 3).mean(axis=0)
                        _tl = _mm0[0:56, 0:56].reshape(-1, 3).mean(axis=0)
                        _cand2 = None
                        if _br[0] > _br[2] * 1.05 and _br[0] > _tl[0]:
                            _cand2 = "blue"
                        elif _tl[2] > _tl[0] * 1.05 and _tl[2] > _br[2]:
                            _cand2 = "red"
                        if _cand2:
                            _votes[_cand2] = _votes.get(_cand2, 0) + 1
                    except Exception:
                        pass
                    if _cand is None:
                        from wzry.vision.camp import detect_camp_from_center
                        _cand3 = detect_camp_from_center(frame)
                        if _cand3:
                            _votes[_cand3] = _votes.get(_cand3, 0) + 1
                _bv = int(_votes.get("blue", 0))
                _rv = int(_votes.get("red", 0))
                if max(_bv, _rv) >= 4 and _bv != _rv:
                    camp = "blue" if _bv > _rv else "red"
                    cooldowns["camp"] = camp
                    cooldowns.pop("camp_votes", None)
                    print(f"[{datetime.now():%H:%M:%S}] 阵营判断: {'蓝方' if camp == 'blue' else '红方'}"
                          f"(票 蓝{_bv}/红{_rv}) "
                          f"→ 发育路方向 {LANE_DIR_BLUE if camp == 'blue' else LANE_DIR_RED}")
                else:
                    print(f"[{datetime.now():%H:%M:%S}] 阵营判断: 采样中(蓝{_bv}/红{_rv}), 未锁")

            # ---- 寮灞闃靛璇嗗埆坴2.10歮odlens 璇婚夎嫳闆勭晫闈枃瀛楋紝鍚挓棣椾晶=鎴戞柟---
            # 鐢悗鍙扮嚎绋嬮伩鍏嶉樆濉炰富寰幆涚粨鏋滃瓨鍏cooldowns["roster"]
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
                        prompt = ('List heroes seen above: JSON {"upper": [], "lower": []}')
                        r = _sp.run(
                            [_sys.executable, "-X", "utf8",
                             str(ROOT / "scripts" / "train" / "modlens_ask.py"),
                             str(tmp), prompt],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=180)
                        if r.returncode != 0:
                            print(f"[roster] modlens 澶辫触: {r.stderr[:150]}")
                            return
                        txt = r.stdout.strip()
                        start, end = txt.find("{"), txt.rfind("}")
                        if start < 0 or end < 0:
                            print(f"[roster] 瑙ｆ瀽澶辫触: {txt[:150]}")
                            return
                        import json as _json
                        data = _json.loads(txt[start:end + 1])
                        upper, lower = data.get("upper", []), data.get("lower", [])
                        if "閽熼" in upper:
                            ally, enemy = upper, lower
                        else:
                            ally, enemy = lower, upper
                        cooldowns["roster"] = {"ally": ally, "enemy": enemy,
                                               "self_hero": "閽熼"}
                        print(f"[roster] 鎴戞柟 {ally} | 鏁屾柟 {enemy}")
                    except Exception as e:
                        print(f"[roster] 寮傚父: {e}")
                    finally:
                        cooldowns["roster_pending"] = 0.0

                _th.Thread(target=_roster_worker, args=(_frame_cp,), daemon=True).start()

            # ---- 鎰熺煡坴2.7歒OLO 闄嶉岀紦瀛樺鐢級----
            yolo_count += 1
            if yolo_count >= yolo_every:
                yolo_count = 0
                cached_dets = det.detect(frame)
            dets = list(cached_dets)
            # v2.91 野怪互斥(源头): neutral_monster 框边距20px内重叠的 enemy_hero = 误标野怪 -> 剔除;
            #    低置信 enemy_hero (0.35-0.55) 且无 monster 佐证也有风险, 置信门槛 0.55 放行
            # v2.98 体型上限: 暴君/主宰 box 宽~276px >> 英雄(<=200px); 暴君血条厚 10-12px
            #    (英雄 5-11px). 宽>215 或 高>270 的 enemy_hero = 巨兽(暴君/主宰/大龙), 剔除
            try:
                _mon_b = [(d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3])
                          for d in dets if d.cls == "neutral_monster"]
                if _mon_b:
                    def _near_mon(d):
                        cx, cy = d.center[0], d.center[1]
                        for (mx0, my0, mx1, my1) in _mon_b:
                            if cx >= mx0 - 20 and cx <= mx1 + 20 \
                                    and cy >= my0 - 20 and cy <= my1 + 20:
                                return True
                        return False
                    dets = [d for d in dets
                            if not (d.cls == "enemy_hero" and _near_mon(d))]
                def _not_beast(d):
                    if d.cls != "enemy_hero":
                        return True
                    _bw = d.xyxy[2] - d.xyxy[0]
                    _bh = d.xyxy[3] - d.xyxy[1]
                    return not (_bw > 215 or _bh > 270)
                dets = [d for d in dets if _not_beast(d)]
            except Exception:
                pass
            # hero identity bar-top color guard
            try:
                _hsv0 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                _h0, _s0 = _hsv0[..., 0].astype(int), _hsv0[..., 1].astype(int)
                for _d in dets:
                    if _d.cls not in ("enemy_hero", "ally_hero"):
                        continue
                    x1, y1, x2, y2 = _d.xyxy
                    _cy0 = int(y1); _cy1 = int(min(frame.shape[0], y1 + max(2, (y2 - y1) // 6)))
                    _cx = int((x1 + x2) // 2)
                    if _cy1 - _cy0 < 2 or _cx - 18 < 0 or _cx + 18 >= frame.shape[1]:
                        continue
                    _ph = _h0[_cy0:_cy1, _cx - 18:_cx + 18]
                    _ps = _s0[_cy0:_cy1, _cx - 18:_cx + 18]
                    _nr = int((((_ph <= 15) | (_ph >= 165)) & (_ps > 60)).sum())
                    _nb = int((((_ph >= 90) & (_ph <= 135)) & (_ps > 60)).sum())
                    _side = None
                    if _nb > _nr + 4 and _nb >= 4:
                        _side = "ally"
                    elif _nr > _nb + 4 and _nr >= 4:
                        _side = "enemy"
                    if _side is None:
                        continue
                    # hysteresis: 2-frame same side
                    _bk = (int(_cx // 40), int(_cy0 // 40))
                    _mem = _hero_cls_mem.get(_bk)
                    if _mem and _mem[0] == _side:
                        _hero_cls_mem[_bk] = (_side, _mem[1] + 1)
                        if _mem[1] + 1 >= 2:
                            if _side == "ally" and _d.cls == "enemy_hero":
                                _d.cls = "ally_hero"
                            elif _side == "enemy" and _d.cls == "ally_hero":
                                _d.cls = "enemy_hero"
                    elif _mem is None:
                        _hero_cls_mem[_bk] = (_side, 1)
            except Exception:
                pass
            st = build_state(frame, dets, phase.value, minimap={                "found": mm["found"], "center": mm["center"], "radius": mm["radius"],
                "dots": mm["dots"], "towers": mm["towers"],
            }, frame_id=frame_id)
            frame_id += 1
            st.t = time.time()  # 鍐崇瓥鏃跺埢
            state_dict = st.to_dict()
            # v2.65 鏁岃嫳寮哄埗娉叆 (涓嶄緷璧build/filter: dets 绾殑鏁岃嫳蹇呰繘 units)
            try:
                # v2.91 野怪互斥: neutral_monster 框与 enemy_hero 框重叠(IoU>0.15/中心距<40px)
                #     -> 该 enemy_hero 实为野怪(yolo 常把野猪/龙误标 enemy_hero 0.35-0.55)
                _mon_b = []
                for _d in dets:
                    if _d.cls == "neutral_monster":
                        _mon_b.append((_d.xyxy[0], _d.xyxy[1], _d.xyxy[2], _d.xyxy[3]))
                def _overlaps_monster(_d):
                    if not _mon_b:
                        return False
                    _cx, _cy = _d.center[0], _d.center[1]
                    _w2, _h2 = (_d.xyxy[2] - _d.xyxy[0]) / 2.0 + 15, (_d.xyxy[3] - _d.xyxy[1]) / 2.0 + 15
                    for (mx0, my0, mx1, my1) in _mon_b:
                        if abs(_cx - (mx0 + mx1) / 2) < _w2 + (mx1 - mx0) / 2 \
                                and abs(_cy - (my0 + my1) / 2) < _h2 + (my1 - my0) / 2:
                            return True
                    return False
                _de = [d for d in dets if d.cls == "enemy_hero"
                       and d.conf >= 0.55        # v2.91 敌英置信下限(野怪误标通常 0.35-0.52)
                       and not _overlaps_monster(d)]
                if _de and not any(u.get("cls") == "enemy_hero" for u in state_dict["units"]):
                    for _d in _de:
                        state_dict["units"].append({
                            "cls": "enemy_hero",
                            "screen": [_d.center[0] / w, _d.center[1] / h,
                                       (_d.xyxy[2] - _d.xyxy[0]) / w,
                                       (_d.xyxy[3] - _d.xyxy[1]) / h]})
                # v2.78 鏁岃嫳蹇呴』甯孩鏉>=5px): None(鏃犳潯)=閲庢璇 -> 鍓旈櫎(鎵撹嚜瀹堕噹鎬墸鍒嗗皝鍫
                from wzry.vision.self_bars import hero_bar_check
                _kept_u = []
                for _u in state_dict.get("units") or []:
                    if _u.get("cls") != "enemy_hero":
                        _kept_u.append(_u)
                        continue
                    _s = _u.get("screen") or [0.5, 0.5, 0.05, 0.05]
                    _cx = float(_s[0]) * w
                    _cy = (float(_s[1]) + float(_s[3]) * 0.6) * h
                    _bh = hero_bar_check(frame, _cx, _cy)
                    if _bh is None or _bh < 5:
                        continue          # 鏃犳潯/钖勬潯=閲庢垨璇, 鍓旈櫎(涓嶆墦浜
                    if _bh >= 9:
                        _kept_u.append(_u)   # 条厚>=9px=英雄(野怪条通常<=8px, 但龙/暴君更粗
                    else:
                        # v2.91 薄条(5-8px)需二次证据: 小地图同侧有红点或 yolo 置信>=0.85
                        _mm_red_u = ((mm.get("dots") or {}).get("red") or [])
                        _hi_c = any(
                            (abs(d.center[0] / w - _s[0]) < 0.30 and
                             abs(d.center[1] / h - _s[1]) < 0.30 and d.conf >= 0.85)
                            for d in dets if d.cls == "enemy_hero")
                        if _hi_c or _mm_red_u:
                            _kept_u.append(_u)
                state_dict["units"] = _kept_u
            except Exception:
                pass
            # ---- 鐞嗚灞傚寮猴紙v2.12夛細鑷繁HP/MP + 鏁屼汉绾潯 + 闃熷弸鏉+ 闃靛 ----
            if now - float(cooldowns.get("hp_t", 0.0)) >= 0.5:
                cooldowns["hp_t"] = now
                try:
                    from wzry.vision.self_bars import self_hp_mp, detect_all_bars, death_banner_px, kill_banner_px
                    # v2.87 死亡横幅铁证 -> ui.death_banner (行为层判死依据)
                    # v2.96 击杀播报金条 -> ui.kill_banner (kill/assist 铁证, 旧"消失=击杀"为假事件)
                    _ui = state_dict.setdefault("ui", {})
                    _ui["death_banner"] = death_banner_px(frame)
                    _ui["kill_banner"] = kill_banner_px(frame)
                    hp, mp, hero_pos = self_hp_mp(frame)
                    if hp is not None:
                        state_dict.setdefault("ui", {})["hp"] = hp
                    if mp is not None:
                        state_dict.setdefault("ui", {})["mp"] = mp
                    if hero_pos:
                        state_dict.setdefault("extra", {})["hero_pos"] = list(hero_pos)
                    # v2.37 绾钃濇潯妫娴嬮檷棰韬唤鍒畾宸蹭互忓湴鍥句负璇 鏉娴嬩粎杈呭姪杩藉嚮)
                    if now - float(cooldowns.get("bars_t", 0.0)) > 0.4:
                        bars = detect_all_bars(frame)
                        cooldowns["bars_t"] = now
                        if bars["enemies"]:
                            state_dict.setdefault("extra", {})["enemy_bars"] = [
                                (b["x"], b["y"], b["w"]) for b in bars["enemies"]]
                        if bars["allies"]:
                            state_dict.setdefault("extra", {})["ally_bars"] = [
                                (b["x"], b["y"], b["w"]) for b in bars["allies"]]
                        state_dict.setdefault("extra", {})["ally_bars"] = [
                            (b["x"], b["y"], b["w"]) for b in bars["allies"]]
                except Exception:
                    pass
                # 鎶鑳界姸鎬侊紙鏍噯鍧愭爣                try:
                    from wzry.vision.ui_reader import skill_ready_state
                    ss = skill_ready_state(frame, skills_pts)
                    state_dict.setdefault("ui", {})["skill_states"] = ss
                except Exception:
                    pass
                # 闃靛堝宸茶瘑鍒級
                if cooldowns.get("roster"):
                    state_dict.setdefault("extra", {})["roster"] = cooldowns["roster"]
            n_ticks += 1
            infer_sum += det.last_infer_ms

            # ---- 瑙勫垯鍐崇瓥 + v2.40 鏈浼樺喅绛栧櫒(鏋氫妇鍊欓>浠峰艰瘎浼>閫夋渶浼 <2ms) ----
            action = decide(state_dict, cooldowns)
            try:
                from wzry.decision.optimizer import solve as _solve
                _hp0 = float((state_dict.get("ui") or {}).get("hp") or 1.0)
                _camp0 = cooldowns.get("camp") or "blue"
                action = _solve(state_dict, action, _hp0, _camp0)
            except Exception as _oe:
                pass  # 浼樺寲鍣紓甯镐笉闃绘柇(淇濆簳=瑙勫垯鍐崇瓥)
            sig = (action.get("type"), action.get("id"))
            if sig != last_sig:
                print(f"[{datetime.now():%H:%M:%S}] [鍐崇瓥] {format_decision(action)}")
                last_sig = sig

            # ---- v3.0 鐞嗚灞傦細琛屼负鏍囩 + 鍙嶉灞傚鎯----
            beh = tagger.update(state_dict, cooldowns, action, now)
            state_dict["behavior"] = beh
            # v2.69 浜嬩欢鍘婚噸: 鍚屼竴浜嬩欢 30s/10s 鑺傛祦(姝墠澶氬抚閲嶅瑙彂閫犳垚'10鍔敾'鍋囪处)
            _evgate = {"assist": 30.0, "kill": 30.0, "hook": 15.0,
                       "died": 6.0, "be_tower_attacked": 8.0,
                       "tower_kill": 10.0, "minion_clear": 6.0}
            _ev = beh.get("event")
            if _ev:
                _lg = _evgate.get(_ev, 10.0)
                _lt = float(cooldowns.get(f"evt_{_ev}_t", 0.0))
                if _lt and now - _lt < _lg:
                    _ev = None
                else:
                    cooldowns[f"evt_{_ev}_t"] = now
            if _ev:
                sc = reward.on_event(_ev)
                if sc:
                    print(f"[{datetime.now():%H:%M:%S}] [反馈] 事件 {_ev} "
                          f"{sc:+.0f} 分（总分 {reward.total:.0f}）")
                # online learning
                try:
                    _hp_now = float((state_dict.get("ui") or {}).get("hp") or 1.0)
                    _mm_now = (state_dict.get("minimap") or {}).get("dots") or {}
                    _n_enemy = len(_mm_now.get("red") or [])
                    _n_ally = len(_mm_now.get("blue") or [])
                    _w_before = dict(learner.weights())
                    learner.observe(_ev, near_enemies=_n_enemy,
                                    hp_low=_hp_now < 0.4, ally_near=_n_ally > 0)
                    _w_after = learner.weights()
                    _delta = " ".join(f"{k}:{_w_before[k]:.2f}->{_w_after[k]:.2f}"
                                      for k in _w_after
                                      if abs(_w_before[k] - _w_after[k]) > 1e-3)
                    if _delta:
                        print(f"[学习] {_ev} | {_delta} | "
                              f"累计 k/a/d={learner.stats().get('kill',0)}/"
                              f"{learner.stats().get('assist',0)}/{learner.stats().get('died',0)}")
                    from wzry.decision.optimizer import refresh_weights as _rw
                    _rw(_w_after)
                except Exception:
                    pass
            if now - float(cooldowns.get("dump_t", 0.0)) > 60.0:
                # learning log rolling 60s
                cooldowns["dump_t"] = now
                try:
                    if learner is not None:
                        _dk = learner.dump_log(str(ROOT / "docs" / "LEARNING_LOG.md"))
                        if _dk:
                            print(f"[{datetime.now():%H:%M:%S}] [学习] 日志滚动已写入 "
                                  f"docs/LEARNING_LOG.md")
                except Exception:
                    pass
            if beh.get("dead"):
                # 闃典骸氱瓑寰呮硥姘村娲伙紝涓嶅仛浠讳綍鎿嶄綔堢敤鎴疯鍒欙級
                action = {"type": "none", "reason": "dead_waiting"}
                sig = (action.get("type"), action.get("id"))
            if last_dead and not beh.get("dead"):
                cooldowns["revive_t"] = now   # v2.49 璁板綍澶嶆椿鏃跺埢(5s 鍐呬笉鎺垬)
            last_dead = bool(beh.get("dead"))

            # ---- v2.43 琚鎵撹嚜鐪 鑷繁(绱湀)鍦晫濉旀敾鍑诲湀鍐-> 寮哄埗鑴辩 ----
            try:
                _hp_v = (state_dict.get("extra") or {}).get("hero_pos")
                if _hp_v and action.get("type") not in ("recall", "restore", "skill"):
                    _hx, _hy = _hp_v[0] / w, _hp_v[1] / h
                    for _tu in (state_dict.get("units") or []):
                        if str(_tu.get("cls")) != "enemy_turret":
                            continue
                        _ts = _tu.get("screen") or [0.5, 0.5]
                        _td = math.hypot(_ts[0] - _hx, (_ts[1] - _hy) * aspect)
                        if _td < 0.24:
                            _ax, _ay = _away_map(state_dict, _ts[0], _ts[1])
                            action = {"type": "map_move", "nx": _ax, "ny": _ay,
                                      "reason": "under_tower_fire"}
                            sig = ("map_move", None)
                            break
            except Exception:
                pass

            # ---- 鎾炲鎰熺煡 v2.64堝惈 map_move: 鐐瑰嚮瀵昏矾涔熷湪绉诲姩, 鍗闇缁曡---
            if action.get("type") in ("move", "map_move"):
                # v2.8氬繁鏂逛綅缃= 忓湴鍥剧豢鑹茬偣堢敤鎴疯涔夛細缁垮湀=鑷繁                hero_pos = None
                if mm.get("found"):
                    greens = (mm.get("dots") or {}).get("green") or []
                    if greens:
                        hero_pos = min(greens, key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
                    else:
                        # 缁胯壊鐐规湭妫鍑烘椂閫鍖栦负钃壊鐐癸紙闃熷弸杩戜技屽瀹癸級
                        blues = (mm.get("dots") or {}).get("blue") or []
                        if blues:
                            hero_pos = min(blues, key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
                wall_hit = wall_sensor.update(now, True, hero_pos)
                if wall_hit:
                    # v2.64 map_move 鍗: 鐩爣鐐瑰悜鍙充晶鍋忕閲嶅彂(浜浛渚
                    if action.get("type") == "map_move":
                        _side = 1 if (wall_sensor._bypass if hasattr(wall_sensor, "_bypass") else 1) else -1
                        try:
                            wall_sensor._bypass = not getattr(wall_sensor, "_bypass", True)
                        except Exception:
                            pass
                        _nx = max(0.02, min(0.98, float(action.get("nx", 0.5)) + 0.06 * _side))
                        _ny = max(0.02, min(0.98, float(action.get("ny", 0.5)) + 0.03))
                        action = {"type": "map_move", "nx": _nx, "ny": _ny,
                                  "reason": "wall_avoid_map"}
                        print(f"[{datetime.now():%H:%M:%S}] [撞墙] 目标点旁移 0.06 -> 重发")
                        sig = ("map_move", None)
                    else:
                        action = wall_sensor.avoid_action(float(action.get("theta", 0.0)))
                        print(f"[{datetime.now():%H:%M:%S}] [撞墙] 位置未动 -> 绕行")
                        sig = ("move", None)
            else:
                wall_sensor.update(now, False, None)

            # ---- 閽瓙勭娴嬮噺璁板綍堣嚜鍔爣瀹氫簩鎶鑳借寖鍥达級----
            if action.get("type") == "skill" and action.get("id") == 2:
                _rec_hook_measure(cooldowns, state_dict, action, hit=False)
            if action.get("type") == "combo":
                _rec_hook_measure(cooldowns, state_dict, action, hit=True)

            # ---- 鎵 / 妯嫙 ----
            if action.get("type") != "none":
                if do_action:
                    try:
                        apply_action(ex, action, cooldowns)
                        n_actions += 1
                    except Exception as _e:
                        print(f"[exec] 鎵寮傚父(璺宠繃): {_e}")
                    # continuous stick: move 0.55s; instant actions release
                    if action.get("type") == "move":
                        _last_move[:] = [action.get("theta"), action.get("r", MOVE_R),
                                         now + max(0.55, action.get("duration_ms", 380) / 1000.0)]
                    elif action.get("type") in ("recall", "restore", "summoner"):
                        _last_move[:] = [None, None, 0.0]
                else:
                    pass   # simulation mode: no execution

            # ---- capture ----
            if args.save:
                if not recorder.active:
                    recorder.start(meta={"agent": "m2_agent_v2", "model": str(args.model),
                                         "action": "on" if do_action else "off"})
                # v2.10氶樀瀹瑰啓鍏姸鎬佹祦堝喅绛栧眰鍙                if cooldowns.get("roster"):
                    state_dict.setdefault("extra", {})["roster"] = cooldowns["roster"]
                recorder.on_state(state_dict)
                if do_action and action.get("type") != "none":
                    rec = dict(action)
                    rec["t"] = time.time()
                    if rec.get("type") == "combo":
                        # 杩炴嫑鎷嗘垚瀛愬姩浣滃瓨妗ｏ紙encode_action 鍏煎                        for sub in rec.get("actions", []):
                            sub_rec = dict(sub)
                            sub_rec["t"] = rec["t"]
                            sub_rec["reason"] = rec.get("reason", "hook_confirmed")
                            recorder.on_action(sub_rec)
                    else:
                        recorder.on_action(rec)

            # ---- v2.15 实时分析标注落盘(temp/live_annot/ 窗口同源显示) ----
            try:
                vis_out = render_live_analysis(frame, dets, mm, state_dict, action, beh,
                                               reward.total, now)
            except Exception as e:
                print(f"[annot] 渲染异常: {e}")
                vis_out = frame.copy()

            # v2.81 self-diagnose: 带标注帧 + 血条读数 一起送 AI 审检
            try:
                if self_check is not None:
                    self_check.feed(vis_out, dets,
                                    ui=(state_dict.get("ui") or {}))
            except Exception:
                pass
            # v2.83 局内行为习惯学习(对面/队友/自己)
            try:
                if behavior_miner is not None:
                    behavior_miner.feed(mm.get("dots") or {})
            except Exception:
                pass

            # ---- 鍛湡鏃織 / 棰勮 ----
            if now - last_log >= 0.5:
                last_log = now
                objs = ", ".join(f"{d.cls}:{d.conf:.2f}" for d in dets[:5]) or "none"
                mm_txt = (f"map blue {len(mm['dots']['blue'])}/red {len(mm['dots']['red'])} "
                          f"({tracker.last_ms:.0f}ms)") if mm["found"] else "map not found"
                print(f"[{datetime.now():%H:%M:%S}] 检测 {det.last_infer_ms:5.0f}ms | "
                      f"{objs} | {mm_txt} | 行为: {beh.get('label', '')} "
                      f"| 总分: {reward.total:.0f} | 决策: {format_decision(action)}")
            if args.show:
                # v2.22 绐楀彛涓庤惤鐩樺悓婧 鍏绱犳爣娉鏂瑰舰忓湴鍥鹃潰鏉涓枃+琛屼负鏉, 涓嶅啀鐢棫鍦嗙洏
                try:
                    from PIL import ImageFont  # noqa: F401
                except Exception:
                    pass
                vis = vis_out if vis_out is not None else frame.copy()
                cv2.imshow("m2-agent-v2", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(f"\n完成: {n_frames}帧 感知{n_ticks} 次 "
              f"平均检测 {infer_sum / max(1, n_ticks):.0f}ms "
              f"动作 {n_actions} 次 模式: {'执行' if do_action else '模拟'}")
    except KeyboardInterrupt:
        print("\n手动退出。")
    finally:
        if recorder.active:
            recorder.close()
        cap.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
