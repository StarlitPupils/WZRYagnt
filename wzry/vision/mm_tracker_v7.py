# -*- coding: utf-8 -*-
"""小地图时序跟踪器 v7：包装 MMDetectorV7 增加时间维。

解决的问题（单帧规则无法完美处理的）：
  1) 英雄环弧段分身：同一环被邻环/地图切开成多个候选（相距 6-24px）。
     时序上只有"环心强候选"能持续产生稳定轨迹，弧段是边缘碎片：
     - 候选匹配到已有轨迹(<=13px) → 吸收进轨迹（EMA 更新）
     - 未匹配候选若距任一已有轨迹 <=24px → 视为碎片丢弃（除非环带分>=0.52 强环心）
  2) 小兵点伪点（地图艺术静态亮斑）：伪点不动，真兵点沿兵线移动：
     输出要求 年龄>=3 帧 且 累计位移 >=3px。

输出兼容旧接口（dots: blue/red/yellow/green + towers 列表），
同时保留新键（self/ally/enemy/monster/buff/minions 词典）。
"""
import math
import time

import numpy as np

from wzry.vision.mm_rules_v7 import MMDetectorV7

HERO_SPEED = 13.0      # 英雄候选-轨迹匹配 px/帧
FRAG_DIST = 24.0       # 碎片抑制半径
STRONG_RING = 0.52     # 强环心（允许在碎片半径内开新轨迹）
MINION_SPEED = 12.0    # 小兵点匹配 px/帧
MINION_AGE = 3
MINION_MOVE = 3.0      # 累计位移阈值 px


class _Track:
    __slots__ = ("x", "y", "age", "miss", "conf", "init_x", "init_y", "dead", "hist")

    def __init__(self, x, y, conf):
        self.x, self.y = float(x), float(y)
        self.age = 1
        self.miss = 0
        self.conf = conf
        self.init_x, self.init_y = float(x), float(y)
        self.dead = False
        self.hist = [(float(x), float(y))]   # v2.39 位置历史(中值滤波用, 最近3帧)


class MMTrackerV7:
    def __init__(self):
        self.det = MMDetectorV7()
        self._mm_cache_f = None   # v13.6 mm检测缓存
        self._mm_cache_t = 0.0
        self._last_mm = None   # v2.90 最近小地图裁剪(输出点重闸用)
        self._last_self = None # v3.3 上帧自己位置(缓慢移动闸)
        self.tracks = {"self": {}, "ally": {}, "enemy": {}}
        self.minion_tracks = {"ally": {}, "enemy": {}}
        self.frame_no = 0
        self.last_ms = 0.0

    # ---------- 英雄 ----------
    def _update_heroes(self, cls, cands):
        tr = self.tracks[cls]
        # 现有轨迹 miss++
        for t in tr.values():
            t.miss += 1
        used = set()
        for (cx, cy, score) in cands:
            best_id, best_d = None, HERO_SPEED
            for tid, t in tr.items():
                if tid in used or t.dead:
                    continue
                d = ((cx - t.x) ** 2 + (cy - t.y) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is not None:
                t = tr[best_id]
                # v2.39 跳变守卫: 候选与本轨迹差>16px 且轨迹已有历史 -> 视为假候选, 不吸收
                if len(t.hist) >= 2 and ((cx - t.x) ** 2 + (cy - t.y) ** 2) ** 0.5 > 16:
                    continue
                t.x = t.x * 0.55 + cx * 0.45
                t.y = t.y * 0.55 + cy * 0.45
                t.conf = max(t.conf, score)
                t.miss = 0
                t.age += 1
                t.hist.append((t.x, t.y))
                t.hist = t.hist[-3:]
                used.add(best_id)
                continue
            # 新轨迹：碎片抑制（距任一轨迹<=24px 的弱候选=弧段）
            near_frag = False
            for t in tr.values():
                if t.dead:
                    continue
                if ((cx - t.x) ** 2 + (cy - t.y) ** 2) ** 0.5 <= FRAG_DIST:
                    near_frag = True
                    break
            if near_frag and score < STRONG_RING:
                continue
            if not tr:
                nid = 0
            else:
                nid = max(tr) + 1
            tr[nid] = _Track(cx, cy, score)
        # 清理长丢轨迹
        for tid in [tid for tid, t in tr.items() if t.dead or t.miss > 12]:
            del tr[tid]

    def _hero_result(self, cls):
        out = []
        for t in self.tracks[cls].values():
            if t.miss <= 1 and t.age >= 2:   # v2.58 miss<=1: 1帧漏检不闪烁
                # v2.45 自己: 低置信(假自环)不输出
                if cls == "self" and t.conf < 0.40:
                    continue
                # v2.87 红点死守防线: 敌方轨迹必须"动" (幽灵=塔饰/角饰红环, 完全静止)
                # 位置历史跨度 <1.5px 且年龄>=6帧 -> 静物装饰, 丢弃; 真实敌方位移明显
                if cls == "enemy" and t.age >= 6 and len(t.hist) >= 3:
                    xs = [p[0] for p in t.hist]
                    ys = [p[1] for p in t.hist]
                    if max(xs) - min(xs) < 1.5 and max(ys) - min(ys) < 1.5:
                        continue
                # v2.90 输出点重闸: 平滑后的轨迹点必须仍为敌英(头像)模板
                # (原始检测点过闸, 平滑输出点可能漂离头像 -> 蓝环队友误入红点)
                if cls == "enemy" and self._last_mm is not None:
                    try:
                        from wzry.vision.mm_rules_v7 import _has_avatar
                        if not _has_avatar(self._last_mm, int(round(t.x)),
                                           int(round(t.y)), thr=0.62):
                            continue
                    except Exception:
                        pass
                # v2.39 位置中值滤波: 输出=最近3帧中值(消除单帧漂移闪烁)
                if len(t.hist) >= 3:
                    hx = sorted(p[0] for p in t.hist)[1]
                    hy = sorted(p[1] for p in t.hist)[1]
                    out.append((hx, hy, t.conf))
                else:
                    out.append((t.x, t.y, t.conf))
        if cls == "self" and out:
            # v3.3 用户铁律: 自己位置只做缓慢小幅移动, 大幅跳跃=误检 -> 输出前与上帧自位对比
            #   self_jump_max_px: 小地图232px, 英雄移速极慢; 单帧(6Hz)跳跃>25px 即误检
            out = sorted(out, key=lambda p: -p[2])[:1]
            if self._last_self is not None:
                _dd = math.hypot(out[0][0] - self._last_self[0],
                                 out[0][1] - self._last_self[1])
                if _dd > 25.0:
                    # 跳变: 保留旧位(预测=旧位), 不污染
                    out[0] = (self._last_self[0], self._last_self[1], 
                              min(out[0][2], 0.35))
        elif cls == "ally" and out:
            # v2.45 我方英雄稳定: ① 不与"自己"重叠(>0.03) ② 两两高并(>0.02 距)
            selfs = self._hero_result_inner("self")
            if selfs:
                sx, sy, _ = selfs[0]
                out = [p for p in out if math.hypot(p[0] - sx, p[1] - sy) > 0.03]
            merged = []
            for p in sorted(out, key=lambda q: -q[2]):
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.02 for q in merged):
                    continue
                merged.append(p)
            out = merged
        return out

    def _hero_result_inner(self, cls):
        out = []
        for t in self.tracks[cls].values():
            if t.miss <= 1 and t.age >= 2:
                out.append((t.x, t.y, t.conf))
        return out

    # ---------- 小兵点 ----------
    def _update_minions(self, cls, cands):
        tr = self.minion_tracks[cls]
        for t in tr.values():
            t.miss += 1
        used = set()
        for (cx, cy) in cands:
            used_d = MINION_SPEED
            best_id, best_d = None, used_d
            for tid, t in tr.items():
                if tid in used or t.dead:
                    continue
                d = ((cx - t.x) ** 2 + (cy - t.y) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is not None:
                t = tr[best_id]
                t.x = t.x * 0.5 + cx * 0.5
                t.y = t.y * 0.5 + cy * 0.5
                t.miss = 0
                t.age += 1
                used.add(best_id)
            else:
                nid = max(tr) + 1 if tr else 0
                tr[nid] = _Track(cx, cy, 0.5)
        for tid in [tid for tid, t in tr.items() if t.dead or t.miss > 5]:
            del tr[tid]

    def _minion_result(self, cls):
        out = []
        for t in self.minion_tracks[cls].values():
            if t.age >= MINION_AGE and t.miss == 0:
                move = ((t.x - t.init_x) ** 2 + (t.y - t.init_y) ** 2) ** 0.5
                if move >= MINION_MOVE:
                    out.append((t.x, t.y, 0.5))
        return out

    # ---------- 主入口 ----------
    def update(self, frame):
        t0 = time.perf_counter()
        # v13.6 mm.detect 200ms缓存(小地图点200ms内不变, 省137ms/帧, 功能不变)
        _nwc = time.perf_counter()
        r = self._mm_cache_f if (self._mm_cache_f is not None and _nwc - self._mm_cache_t < 0.2) else None
        if r is None:
            r = self.det.detect(frame)
            self._mm_cache_f = r
            self._mm_cache_t = _nwc
        # v2.90 输出点重闸: 留存小地图裁剪 (0..msz 方区, 与 det 同坐标系)
        r0 = r.get("size", 232) or 232
        self._last_mm = (frame[0:int(r0), 0:int(r0)]
                         if frame.shape[0] >= int(r0) and frame.shape[1] >= int(r0) else None)
        if not r.get("found"):
            # v2.27 非对局: 清空轨迹并返回未找到(避免大厅/选英雄伪标注与伪决策)
            for cls in ("self", "ally", "enemy"):
                self.tracks[cls] = {}
                self.minion_tracks[cls] = {}
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            return {"found": False, "center": None, "radius": None, "method": "v7-gate",
                    "dots": {"blue": [], "red": [], "green": [], "yellow": []},
                    "towers": [], "minions": [], "v7": {"self": [], "ally": [], "enemy": [],
                                                        "monster": [], "buff": [],
                                                        "minions": {"ally": [], "enemy": []}},
                    "towers_v7": {"ally": [], "enemy": []}}
        msz = float(r.get("size", 232))
        nx = lambda p: p[0] / msz
        ny = lambda p: p[1] / msz

        # 英雄时序
        for cls in ("self", "ally", "enemy"):
            cands = [(d["n"][0] * msz, d["n"][1] * msz, float(d.get("conf", 0.5)))
                     for d in r["dots"][cls]]
            self._update_heroes(cls, cands)
        heroes = {cls: self._hero_result(cls) for cls in ("self", "ally", "enemy")}
        if heroes["self"]:
            self._last_self = (heroes["self"][0][0], heroes["self"][0][1])
        # else: 保留旧 _last_self (漏检帧不清零, 防止新位置突然出现被放行)

        # 小兵点时序（伪点静止过滤）
        for cls in ("ally", "enemy"):
            self._update_minions(cls, [(d["n"][0] * msz, d["n"][1] * msz) for d in r["minions"][cls]])
        minions = {cls: self._minion_result(cls) for cls in ("ally", "enemy")}

        # ---- 兼容旧接口: 蓝=队友 红=敌人 绿=自己 黄=野怪 ----
        # v3.3 用户铁律: 5v5 上限 = 敌英最多5, 队友最多4(自己1) -> 按置信度截断
        heroes_enemy = sorted(heroes["enemy"], key=lambda p: -p[2])[:5]
        heroes_ally = sorted(heroes["ally"], key=lambda p: -p[2])[:4]
        heroes_self = heroes["self"][:1]
        # v4.1 ROI 分类器门控: 小地图候选抠图过训练分类器, 类目不符剔除
        #   (mm_red才算敌英/mm_blue才算队友/mm_green才算自己; 假红点/假蓝点被纠正)
        try:
            from wzry.vision.roi_cls_gate import filter_dots
            _mm_crop = frame[0:int(msz), 0:int(msz)]
            _r0, _b0, _g0 = filter_dots(
                _mm_crop,
                [(p[0] / msz, p[1] / msz, p[2]) for p in heroes_enemy],
                [(p[0] / msz, p[1] / msz, p[2]) for p in heroes_ally],
                [(p[0] / msz, p[1] / msz, p[2]) for p in heroes_self],
                size=int(msz))
            heroes_enemy = [(p[0] * msz, p[1] * msz, p[2]) for p in _r0]
            heroes_ally = [(p[0] * msz, p[1] * msz, p[2]) for p in _b0]
            heroes_self = [(p[0] * msz, p[1] * msz, p[2]) for p in _g0]
        except Exception:
            pass
        legacy_dots = {
            "blue": [[p[0] / msz, p[1] / msz] for p in heroes_ally],
            "red": [[p[0] / msz, p[1] / msz] for p in heroes_enemy],
            "green": [[p[0] / msz, p[1] / msz] for p in heroes_self],
            "yellow": [[d["n"][0], d["n"][1]] for d in r["dots"]["monster"]],
        }
        # 塔: 旧=[[nx,ny]...], 新=towers{ally,enemy}
        towers_legacy = ([[t["n"][0], t["n"][1]] for t in r["towers"]["ally"]]
                         + [[t["n"][0], t["n"][1]] for t in r["towers"]["enemy"]])
        minions_legacy = ([[p[0] / msz, p[1] / msz] for p in minions["ally"]]
                          + [[p[0] / msz, p[1] / msz] for p in minions["enemy"]])

        def rec(pts):
            return [{"n": [round(px / msz, 4), round(py / msz, 4)],
                     "conf": round(c, 2), "src": "v7tr"} for px, py, c in pts]

        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "found": True,
            "center": r["center"], "radius": r["radius"],
            "method": "v7-track",
            "dots": legacy_dots,
            "towers": towers_legacy,
            "minions": minions_legacy,
            # 新键
            "v7": {"self": rec(heroes["self"]), "ally": rec(heroes["ally"]),
                   "enemy": rec(heroes["enemy"])[:5],
                   "monster": r["dots"]["monster"], "buff": r.get("buff") or [],
                   "minions": {"ally": rec(minions["ally"]), "enemy": rec(minions["enemy"])}},
            "towers_v7": r["towers"],
        }
