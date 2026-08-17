# -*- coding: utf-8 -*-
"""动作反推器 v2（M3 数据工厂核心）：从带 HUD 的对局录像推断玩家动作。

多通道设计（每个通道是独立信号源，最后交叉验证融合）：
  A 摇杆箭头   —— 白色箭头区域（HSV 亮度 + 形态学 + 面积/距离过滤）→ 移动方向（连续角 + 8 向标签）
  B 技能按钮   —— 技能按钮 ROI 亮度相对基线突变（自适应基线，按下=变暗沿）→ 按下事件（时间点）
  C 特效/位移  —— YOLO 11 类模型：skill_effect 检测 + ally_hero 位移方向/突变 + 血条红色占比突变 → 技能释放证据
  D 瞄准线     —— 技能拖动时的亮色射线（HoughLinesP，按钮近端约束）→ 拖瞄事件

交叉验证与置信度：
  移动：A 为主、C（位移方向）为辅 —— 方向一致 → conf 高；冲突 → conf 低并标记 A_C_conflict。
  技能：B 给出按下时间，C 提供特效/位移证据 —— 两者都有 → conf 高。
  输出每条动作 {"t": 秒, "type": "move|skill|attack|none", "direction"/"skill_id",
                "confidence": 0-1, "channels": ["A","C"], "frame": 真实帧号, "flags": [...]}，
  返回列表末尾追加一条 {"type": "meta", "events": [...]} 携带原始事件列表。

接口：
  from wzry.train.action_infer import infer_actions
  actions = infer_actions("xxx.mp4", calib_dict_or_path, sample_every=1, model_path=None)

局限：
  - 通道 A/B/D 依赖 HUD（摇杆/技能按钮），无 HUD 录像（如 temp/tmphjhl7fk3.mp4 自由视角）
    无法真验 → 报告标注"待带 HUD 录像真验"。
  - 通道 C 需要 YOLO 模型（默认 runs/detect/zhongkui_11cls/weights/best.pt），CPU 上较慢，
    可用 yolo_every 稀疏运行。
"""
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 常量

# 屏幕坐标 8 向向量（y 向下：屏幕"上"为负 y）
DIR_VECTORS = {
    "right": (1.0, 0.0),
    "right_up": (0.7071, -0.7071),
    "up": (0.0, -1.0),
    "left_up": (-0.7071, -0.7071),
    "left": (-1.0, 0.0),
    "left_down": (-0.7071, 0.7071),
    "down": (0.0, 1.0),
    "right_down": (0.7071, 0.7071),
}
DIR_ORDER = ["right", "right_down", "down", "left_down",
             "left", "left_up", "up", "right_up"]

CLASSES_11 = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
              "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
              "neutral_monster", "hook_aim", "skill_effect"]

DEFAULT_MODEL = Path("runs/detect/zhongkui_11cls/weights/best.pt")

SKILL_BUTTONS = ["skill1", "skill2", "skill3"]
ALL_BUTTONS = ["skill1", "skill2", "skill3", "attack", "recall", "restore", "summoner"]

# ---------------------------------------------------------------- 工具


def vec_to_dir(dx, dy):
    """屏幕向量 -> 8 向标签（余弦相似度最近）。"""
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    dx, dy = dx / n, dy / n
    best, best_sim = None, -2.0
    for name, (vx, vy) in DIR_VECTORS.items():
        sim = dx * vx + dy * vy
        if sim > best_sim:
            best_sim, best = sim, name
    return best


def angle_diff_deg(a_rad, b_rad):
    """两角（弧度）最小夹角（度）。"""
    d = abs(a_rad - b_rad) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return math.degrees(d)


def clamp01(x):
    return max(0.0, min(1.0, x))


def _disc_mask(shape, center, radius):
    """以 center 为圆心、radius 为半径的圆盘掩码。"""
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2


# ---------------------------------------------------------------- 校准


def load_calib_points(calib, video_size=None):
    """把各种校准格式统一为 (绝对像素点表 dict, [w, h])。

    支持：
      - calibrate_video.py 输出：{"video_resolution": [w,h], "points": {...}}
      - configs/calibration.json（归一化点表）
      - configs/calibration_absolute.json（{"screen_width","screen_height","points"}）
    """
    if isinstance(calib, (str, Path)):
        with open(calib, "r", encoding="utf-8") as f:
            calib = json.load(f)
    if not isinstance(calib, dict):
        raise ValueError("calib 必须是 dict 或校准文件路径")

    w = h = None
    raw_pts = {}
    if "points" in calib:
        raw_pts = dict(calib["points"])
        if "video_resolution" in calib:
            w, h = calib["video_resolution"]
    else:
        for k, v in calib.items():
            if k == "video_resolution" and isinstance(v, (list, tuple)) and len(v) == 2:
                w, h = v
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                raw_pts[k] = v
    if w is None:
        w = int(calib.get("screen_width") or (video_size[0] if video_size else 1280))
    if h is None:
        h = int(calib.get("screen_height") or (video_size[1] if video_size else 720))

    abs_pts = {}
    for k, (x, y) in raw_pts.items():
        x, y = float(x), float(y)
        if max(w, h) > 1.01 and x <= 1.01 and y <= 1.01:
            abs_pts[k] = (x * w, y * h)
        else:
            abs_pts[k] = (x, y)
    # 别名兼容
    if "joystick_center" not in abs_pts and "move_stick_center" in abs_pts:
        abs_pts["joystick_center"] = abs_pts["move_stick_center"]
    return abs_pts, [int(w), int(h)]


# ---------------------------------------------------------------- 通道 A：摇杆箭头


class ArrowChannel:
    """通道 A：白色箭头 -> 移动方向（升级版：圆盘搜索 + 面积/距离过滤 + PCA 尖端估计）。"""

    def __init__(self, pts, r_scale=2.2, min_radius=34):
        self.center = pts.get("joystick_center")
        self.tip = pts.get("joystick_arrow_tip")
        if self.center is None:
            self.enabled = False
            self.radius = 60
        else:
            self.enabled = True
            base = math.hypot(self.tip[0] - self.center[0],
                              self.tip[1] - self.center[1]) if self.tip else 80
            self.radius = max(min_radius, int(r_scale * base))
        self.lower = np.array([0, 0, 190], np.uint8)
        self.upper = np.array([180, 40, 255], np.uint8)

    def detect(self, frame):
        """返回 {"direction","theta","conf"} 或 None。"""
        if not self.enabled:
            return None
        cx, cy = int(self.center[0]), int(self.center[1])
        R = self.radius
        h, w = frame.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        disc = _disc_mask(frame.shape, (cx, cy), R)
        mask = cv2.bitwise_and(mask, mask, mask=disc.astype(np.uint8) * 255)

        # 形态学：开运算去噪，闭运算连接箭头碎片
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        # 面积过滤：按分辨率定阈值（箭头面积不随搜索圆盘面积缩放）
        area_min = max(18.0, 0.00008 * frame_area)
        area_max = min(0.5 * math.pi * R * R, 0.06 * frame_area)
        best = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_min or area > area_max:
                continue
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            bx = m["m10"] / m["m00"]
            by = m["m01"] / m["m00"]
            dist = math.hypot(bx - cx, by - cy)
            # 摇杆底座位于中心：距离过近的 blob 视为底座，过滤
            if dist < 0.12 * R or dist > 1.25 * R:
                continue
            # 长条程度（箭头应为细长形）
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) < 3:
                continue
            aspect = max(rw, rh) / (max(min(rw, rh), 1e-5))
            # 距离权重：箭头尖端应离中心较远（越远越像拖动）
            dist_w = clamp01((dist - 0.12 * R) / (0.9 * R))
            area_w = clamp01(area / max(0.004 * frame_area, 1.0))
            score = 0.45 * dist_w + 0.35 * area_w + 0.20 * clamp01(aspect / 2.5)
            if score > best_score:
                best_score = score
                best = (c, bx, by, area, aspect, dist)

        if best is None:
            return None
        _, bx, by, area, aspect, dist = best

        # 方向：blob 主轴的远端（尖端）指向（比直接用质心更准）
        blob_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(blob_mask, [best[0]], -1, 255, -1)
        ys, xs = np.nonzero(blob_mask)
        if len(xs) < 8:
            return None
        cov = np.cov(np.stack([xs - bx, ys - by]))
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            evals, evecs = np.array([1.0, 1.0]), np.eye(2)
        u = evecs[:, int(np.argmax(evals))]
        if np.dot(u, [bx - cx, by - cy]) < 0:
            u = -u
        span = 3.0 * math.sqrt(max(evals.max(), 4.0))
        tip_x = bx + span * u[0]
        tip_y = by + span * u[1]
        dx, dy = tip_x - cx, tip_y - cy
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        dx, dy = dx / n, dy / n
        theta = math.atan2(dy, dx)

        # 置信度：面积/距离/细长度都在合理区间
        conf = 0.45 + 0.25 * clamp01(area / max(0.004 * frame_area, 1.0)) \
               + 0.15 * clamp01((dist - 0.12 * R) / (0.9 * R)) \
               + 0.15 * clamp01(aspect / 2.5)
        return {"direction": vec_to_dir(dx, dy), "theta": theta,
                "conf": clamp01(conf), "blob_area": area, "dist": dist,
                "tip": (tip_x, tip_y)}


# ---------------------------------------------------------------- 通道 B：按钮高亮


class ButtonChannel:
    """通道 B：按钮 ROI 亮度相对自适应基线突变 -> 按下事件（变暗沿 + 防抖 + 释放锁）。

    真验发现（M2）：王者荣耀按下技能时按钮是**变暗**而非变亮 —— 技能图标被
    按下高亮/冷却遮罩压暗：skill2 正常 ~112 -> 按下瞬间 ~60-90（随后 ~11s 冷却
    遮罩维持 ~60），skill1 正常 ~94.6 -> 按下 ~52-60。旧版按"变亮"检测在本真验
    录像中 8/8 skill2 全漏（唯一一次"检出"是开局白屏覆盖按钮的误报）。

    机制：
      - dark = (base - mean) > delta 且 base > min_base（基线本身要够亮，
        暗色按钮如 attack(~64) 永不触发）
      - 防抖：连续 dark debounce 帧才发射一次按下事件
      - 释放锁：发射后锁定，直到按钮恢复非 dark 且持续 release_s 秒（按
        real_frame 真实帧差计，与采样率无关），防止冷却期遮罩 / 同一次按下
        的双闪被重复触发
      - 基线：仅当 |mean - base| < band 时 EMA 更新 —— 253 全屏亮闪与冷却
        遮罩都不会污染基线
    """

    def __init__(self, pts, r_scale=0.018, delta=20.0, min_base=70.0, debounce=2,
                 ema_alpha=0.05, release_s=1.5, min_interval_s=1.5, band=15.0):
        self.r = max(10, int(r_scale * 1280))
        self.delta = delta
        self.min_base = min_base
        self.debounce = debounce
        self.ema_alpha = ema_alpha
        self.release_s = release_s
        self.min_interval_s = min_interval_s
        self.band = band
        self.buttons = {k: pts[k] for k in ALL_BUTTONS if k in pts}
        self.baseline = {}
        self.streak = {k: 0 for k in self.buttons}
        self.hold = {k: False for k in self.buttons}
        self.normal_run = {k: 0 for k in self.buttons}
        self.last_emit = {k: -1 for k in self.buttons}
        self.last_real_frame = None
        self.seen = 0

    def detect(self, frame, real_frame=None, fps=30.0):
        """返回按下事件列表 [{"button","brightness","baseline","conf"}...]（变暗沿）。

        real_frame/fps 用于把 release_s / min_interval_s 折算成真实帧间隔，
        使锁定时长与抽样率（sample_every）无关。
        """
        events = []
        self.seen += 1
        if real_frame is None:
            real_frame = self.seen
        step = 1 if self.last_real_frame is None else max(1, real_frame - self.last_real_frame)
        self.last_real_frame = real_frame
        release_frames = max(4, int(self.release_s * fps))
        min_gap = max(3, int(self.min_interval_s * fps))
        for name, (x, y) in self.buttons.items():
            xi, yi = int(x), int(y)
            roi = frame[max(0, yi - self.r):yi + self.r,
                        max(0, xi - self.r):xi + self.r]
            if roi.size == 0:
                continue
            mean = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
            if name not in self.baseline:
                self.baseline[name] = mean
                continue
            base = self.baseline[name]
            dark = (base - mean) > self.delta and base > self.min_base
            if dark:
                self.streak[name] += 1
                self.normal_run[name] = 0
            else:
                self.streak[name] = 0
                if abs(mean - base) < self.band:
                    self.baseline[name] = (1 - self.ema_alpha) * base + self.ema_alpha * mean
                if self.hold[name]:
                    self.normal_run[name] += step
                    if self.normal_run[name] >= release_frames:
                        self.hold[name] = False
            if (self.streak[name] >= self.debounce and not self.hold[name]
                    and real_frame - self.last_emit[name] >= min_gap):
                rel = clamp01((base - mean) / max(3 * self.delta, 1e-6))
                events.append({"button": name, "brightness": mean,
                               "baseline": base,
                               "conf": clamp01(0.5 + 0.4 * rel)})
                self.last_emit[name] = real_frame
                self.hold[name] = True
                self.normal_run[name] = 0
                self.streak[name] = 0
        return events


# ---------------------------------------------------------------- 通道 C：特效与位移


class EffectChannel:
    """通道 C：YOLO 11 类模型 —— skill_effect 检测 + ally_hero 位移方向/突变 + 血条突变。"""

    def __init__(self, model_path=None, classes=None, conf_thr=0.3,
                 disp_min=6.0, dash_factor=4.0, hp_drop_thr=0.25, mock_detector=None):
        self.mock_detector = mock_detector
        self.model = None
        self.conf_thr = conf_thr
        self.disp_min = disp_min
        self.dash_thr = disp_min * dash_factor
        self.hp_drop_thr = hp_drop_thr
        self.classes = list(classes) if classes else list(CLASSES_11)
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            names = self.model.names
            if isinstance(names, dict):
                self.classes = [names[i] for i in sorted(names)]
        self.skill_effect_id = self.classes.index("skill_effect") if "skill_effect" in self.classes else -1
        self.hero_id = self.classes.index("ally_hero") if "ally_hero" in self.classes else -1
        self.prev_pos = None
        self.prev_hp_frac = None
        self._det_cache = None  # (real_frame, detections)

    @property
    def enabled(self):
        return self.model is not None or self.mock_detector is not None

    def _run_model(self, frame):
        """返回统一检测列表 [(cls_name, conf, (x1,y1,x2,y2)), ...]。"""
        if self.mock_detector is not None:
            return list(self.mock_detector(frame))
        res = self.model(frame, conf=self.conf_thr, verbose=False)[0]
        out = []
        if res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for box, c, ci in zip(xyxy, confs, clss):
                name = self.classes[ci] if ci < len(self.classes) else str(ci)
                out.append((name, float(c), tuple(float(v) for v in box)))
        return out

    def _hp_bar_red_frac(self, frame, box):
        """英雄血条（box 上方窄条）红色占比。"""
        x1, y1, x2, y2 = (int(v) for v in box)
        bh = max(y2 - y1, 8)
        bw = max(x2 - x1, 8)
        top = max(0, y1 - int(0.14 * bh))
        bot = max(0, y1 - int(0.02 * bh))
        if bot <= top:
            return None
        strip = frame[top:bot, x1 + int(0.08 * bw): x2 - int(0.08 * bw)]
        if strip.size == 0:
            return None
        b, g, r = cv2.split(strip)
        red = ((r > 120) & (r > 1.3 * g) & (r > 1.3 * b)).mean()
        return float(red)

    def detect(self, frame, real_frame):
        """返回 C 事件列表：[{"kind": "skill_effect"|"hero_move"|"dash"|"hp_drop", ...}]。"""
        if not self.enabled:
            return []
        if self._det_cache is not None and self._det_cache[0] == real_frame:
            dets = self._det_cache[1]
        else:
            dets = self._run_model(frame)
            self._det_cache = (real_frame, dets)
        events = []
        hero_pos = None
        hero_box = None
        se_confs = []
        for name, conf, box in dets:
            if name == "skill_effect":
                se_confs.append(conf)
            elif name == "ally_hero" and hero_pos is None:
                hero_pos = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                hero_box = box
        if se_confs:
            events.append({"kind": "skill_effect", "conf": max(se_confs),
                           "n": len(se_confs), "frame": real_frame})

        if hero_pos is not None:
            if self.prev_pos is not None:
                dx = hero_pos[0] - self.prev_pos[0]
                dy = hero_pos[1] - self.prev_pos[1]
                dist = math.hypot(dx, dy)
                if dist >= self.disp_min:
                    events.append({"kind": "hero_move",
                                   "direction": vec_to_dir(dx, dy),
                                   "theta": math.atan2(dy, dx),
                                   "dist": dist,
                                   "conf": clamp01(0.5 + dist / (self.disp_min * 6.0)),
                                   "frame": real_frame})
                    if dist >= self.dash_thr:
                        events.append({"kind": "dash", "dist": dist,
                                       "conf": clamp01(0.6 + dist / (self.dash_thr * 4.0)),
                                       "frame": real_frame})
            self.prev_pos = hero_pos

        if hero_box is not None:
            frac = self._hp_bar_red_frac(frame, hero_box)
            if frac is not None and self.prev_hp_frac is not None \
                    and self.prev_hp_frac - frac > self.hp_drop_thr:
                events.append({"kind": "hp_drop", "frac": frac,
                               "prev_frac": self.prev_hp_frac,
                               "conf": clamp01(0.5 + (self.prev_hp_frac - frac)),
                               "frame": real_frame})
            if frac is not None:
                self.prev_hp_frac = frac
        return events


# ---------------------------------------------------------------- 通道 D：瞄准线


class AimChannel:
    """通道 D：技能拖动时的亮色射线（HoughLinesP + 按钮近端约束）。"""

    def __init__(self, pts, r_scale=0.018):
        self.r = max(10, int(r_scale * 1280))
        skill_pts = [pts[k] for k in SKILL_BUTTONS if k in pts]
        attack_pts = [pts["attack"]] if "attack" in pts else []
        ref = skill_pts or attack_pts
        self.region = None
        if ref:
            xs = [p[0] for p in ref]
            ys = [p[1] for p in ref]
            x0 = max(0, int(min(xs) - 3 * self.r))
            x1 = min(4096, int(max(xs) + 3 * self.r))
            y1 = int(max(ys) + self.r)
            self.region = (x0, 0, x1, y1)
        self.buttons = {k: pts[k] for k in SKILL_BUTTONS if k in pts}

    def detect(self, frame):
        """返回 {"active", "skill", "theta", "conf"} 或 None。"""
        if self.region is None or not self.buttons:
            return None
        x0, y0, x1, y1 = self.region
        h, w = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None
        sub = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, np.array([0, 0, 220], np.uint8),
                             np.array([180, 80, 255], np.uint8))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        min_len = max(36, int(0.18 * (y1 - y0)))
        lines = cv2.HoughLinesP(bright, 1, math.pi / 180, threshold=28,
                                minLineLength=min_len, maxLineGap=14)
        if lines is None:
            return None
        best = None
        best_conf = 0.0
        for ln in lines:
            (ax, ay), (bx, by) = ln[0][:2], ln[0][2:]
            ax, ay = ax + x0, ay + y0
            bx, by = bx + x0, by + y0
            length = math.hypot(ax - bx, ay - by)
            if length < min_len:
                continue
            # 端点须靠近某个技能按钮（瞄准线从按钮拖出）
            near = None
            near_d = 1e9
            for name, (px, py) in self.buttons.items():
                d = min(math.hypot(ax - px, ay - py), math.hypot(bx - px, by - py))
                if d < near_d:
                    near_d, near = d, name
            if near is None or near_d > 2.8 * self.r:
                continue
            conf = clamp01(0.5 + 0.3 * clamp01(length / (3 * min_len))
                           + 0.2 * clamp01(1.0 - near_d / (2.8 * self.r)))
            if conf > best_conf:
                dx, dy = bx - ax, by - ay
                theta = math.atan2(dy, dx)
                best = {"active": True, "skill": near, "theta": theta,
                        "conf": conf, "length": length}
                best_conf = conf
        return best


# ---------------------------------------------------------------- 融合器


class ActionInferrer:
    """组合四通道并融合为动作序列（带状态：基线/前一帧位置/最近 C 事件）。"""

    def __init__(self, pts, video_resolution, model_path=None, c_window_s=0.18,
                 emit_none=False, emit_aim_only=False, mock_detector=None,
                 classes=None):
        self.pts = pts
        self.vid_w, self.vid_h = video_resolution
        self.chA = ArrowChannel(pts)
        self.chB = ButtonChannel(pts)
        self.chC = EffectChannel(model_path=model_path, classes=classes,
                                 mock_detector=mock_detector)
        self.chD = AimChannel(pts)
        self.c_window_s = c_window_s
        self.emit_none = emit_none
        self.emit_aim_only = emit_aim_only
        self.recent_c = deque()      # (t, event)
        self.recent_d = deque()      # (t, aim_event)
        self.recent_a = deque()      # (t, arrow_event)
        self.stats = {"A_arrow": 0, "B_press": 0, "C_skill_effect": 0,
                      "C_hero_move": 0, "C_dash": 0, "C_hp_drop": 0,
                      "D_aim": 0, "move": 0, "skill": 0, "attack": 0,
                      "conflict": 0}

    # -- 原始事件收集（每个采样帧调用） --------------------------------
    def collect_frame(self, frame, real_frame, fps):
        """返回 (frame_events, move_action_or_None)。"""
        t = real_frame / fps
        events = []
        move_act = None

        # A 箭头
        ar = self.chA.detect(frame)
        if ar:
            events.append({"ch": "A", "kind": "arrow", "t": t, "frame": real_frame,
                           "direction": ar["direction"], "theta": ar["theta"],
                           "conf": ar["conf"], "blob_area": ar["blob_area"],
                           "dist": ar["dist"], "tip": ar["tip"]})
            self.recent_a.append((t, ar))
            self.stats["A_arrow"] += 1

        # C 特效/位移
        c_evs = self.chC.detect(frame, real_frame)
        for ev in c_evs:
            ev = dict(ev, ch="C", t=t)
            events.append(ev)
            self.recent_c.append((t, ev))
            kind = ev["kind"]
            self.stats["C_" + kind] = self.stats.get("C_" + kind, 0) + 1

        # D 瞄准线
        aim = self.chD.detect(frame)
        if aim:
            events.append({"ch": "D", "kind": "aim", "t": t, "frame": real_frame,
                           "skill": aim["skill"], "theta": aim["theta"],
                           "conf": aim["conf"], "length": aim["length"]})
            self.recent_d.append((t, aim))
            self.stats["D_aim"] += 1

        # 移动融合：A 主 C 辅
        a_now = self.recent_a[-1][1] if self.recent_a and self.recent_a[-1][0] == t else None
        c_move = [ev for ev in c_evs if ev["kind"] == "hero_move"]
        c_dir = c_move[0] if c_move else None

        if a_now:
            channels = ["A"]
            conf = a_now["conf"]
            flags = []
            if c_dir:
                channels.append("C")
                diff = angle_diff_deg(a_now["theta"], c_dir["theta"])
                if diff <= 45.0:
                    conf = min(0.95, a_now["conf"] + 0.25)
                else:
                    conf = 0.35
                    flags.append("A_C_conflict")
                    self.stats["conflict"] += 1
            move_act = {"t": t, "frame": real_frame, "type": "move",
                        "direction": a_now["direction"], "confidence": conf,
                        "channels": channels, "flags": flags}
            self.stats["move"] += 1
        elif c_dir and c_dir["dist"] >= self.chC.disp_min:
            move_act = {"t": t, "frame": real_frame, "type": "move",
                        "direction": c_dir["direction"], "confidence": 0.5,
                        "channels": ["C"], "flags": ["c_only"]}
            self.stats["move"] += 1

        # B 按钮按下
        for ev in self.chB.detect(frame, real_frame, fps):
            ev = dict(ev, ch="B", kind="press", t=t, frame=real_frame)
            events.append(ev)
            self.stats["B_press"] += 1

        if self.emit_none and move_act is None:
            move_act = {"t": t, "frame": real_frame, "type": "none",
                        "confidence": 0.0, "channels": [], "flags": []}
        return events, move_act

    # -- 技能融合（扫描 B 按下 + C/D 证据窗口） ------------------------
    def _skill_window(self, t):
        """返回 (t 邻域内 C 事件, D aim 事件)。"""
        c_evs = [ev for (tt, ev) in self.recent_c
                 if abs(tt - t) <= self.c_window_s]
        d_evs = [ev for (tt, ev) in self.recent_d
                 if abs(tt - t) <= self.c_window_s]
        return c_evs, d_evs

    def fuse_skills(self, press_events):
        """press_events: 全片 B 按下事件（按 t 排序）；返回技能/攻击动作列表。"""
        actions = []
        claimed_c = set()
        for ev in press_events:
            t = ev["t"]
            c_evs, d_evs = self._skill_window(t)
            btn = ev["button"]
            channels = ["B"]
            conf = ev["conf"]
            flags = []

            if btn.startswith("skill"):
                c_evidence = [c for c in c_evs
                              if c["kind"] in ("skill_effect", "dash", "hp_drop")]
                if c_evidence:
                    channels.append("C")
                    if any(c["kind"] == "skill_effect" for c in c_evidence):
                        conf = min(0.95, conf + 0.35)
                        claimed_c.update(id(c) for c in c_evidence
                                         if c["kind"] == "skill_effect")
                    else:
                        conf = min(0.9, conf + 0.15)
                    if any(c["kind"] == "dash" for c in c_evidence):
                        conf = min(0.95, conf + 0.05)
                aim_match = [d for d in d_evs if d["skill"] == btn]
                if aim_match:
                    channels.append("D")
                    conf = min(0.95, conf + 0.05)
                    flags.append("aim")
                act = {"t": t, "frame": ev["frame"], "type": "skill",
                       "skill_id": int(btn[-1]), "confidence": conf,
                       "channels": channels, "flags": flags}
                self.stats["skill"] += 1
            elif btn == "attack":
                c_evid = [c for c in c_evs if c["kind"] in ("dash", "skill_effect")]
                if c_evid:
                    channels.append("C")
                    conf = min(0.9, conf + 0.2)
                act = {"t": t, "frame": ev["frame"], "type": "attack",
                       "confidence": conf, "channels": channels, "flags": []}
                self.stats["attack"] += 1
            else:
                continue
            actions.append(act)

        # C 特效无 B 确认（可能是敌方技能）—— 低置信度、标记
        for (_ct, c) in self.recent_c:
            if c["kind"] == "skill_effect" and id(c) not in claimed_c:
                actions.append({"t": c["t"], "frame": c.get("frame"),
                                "type": "skill", "skill_id": None,
                                "confidence": 0.6,
                                "channels": ["C"],
                                "flags": ["no_button_confirm", "unconfirmed"]})
                self.stats["skill"] += 1

        # D 拖瞄无 B 按下 —— 拖瞄事件（低置信度、标记）
        if self.emit_aim_only:
            for d in self.recent_d:
                if not any(abs(p["t"] - d["t"]) <= self.c_window_s
                           and p["button"] == d["skill"] for p in press_events):
                    actions.append({"t": d["t"], "type": "skill",
                                    "skill_id": int(d["skill"][-1]),
                                    "confidence": 0.4,
                                    "channels": ["D"],
                                    "flags": ["aim_only", "unconfirmed"]})
                    self.stats["skill"] += 1
        return actions

    def process_video(self, video_path, sample_every=1, yolo_every=1,
                      max_frames=None, debug_viz=None, debug_every=1,
                      progress=True, fps_override=None):
        """主循环：返回 (actions, events, stats)。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")
        fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        actions = []
        events = []
        press_events = []
        frames_done = 0
        debug_dir = Path(debug_viz) if debug_viz else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        read_idx = 0
        while True:
            if max_frames and frames_done >= max_frames:
                break
            if n_frames > 0 and read_idx >= n_frames:
                break
            # 顺序读取（避免 set(POS_FRAMES)+read 对损坏 mkv 的解码不稳定）
            ret, frame = cap.read()
            if not ret:
                break
            # 抽样：按"读取计数"跳过，而不是按 real_frame（旧版 real_frame 恒为
            # sample_every 的倍数，跳过分支永不触发，导致只处理了视频前 1/sample_every）
            if sample_every > 1 and read_idx % sample_every != 0:
                read_idx += 1
                continue
            real_frame = read_idx
            do_yolo = (self.chC.enabled and
                       (frames_done % max(1, yolo_every) == 0))
            if not do_yolo:
                # 跳过 YOLO：用空缓存占位避免跑模型，同时清空位移状态（保守）
                self.chC._det_cache = (real_frame, [])
                self.chC.prev_pos = None
            evs, move_act = self.collect_frame(frame, real_frame, fps)
            events.extend(evs)
            for ev in evs:
                if ev["ch"] == "B":
                    press_events.append(ev)
            if move_act:
                actions.append(move_act)
            if debug_dir and (frames_done % max(1, debug_every) == 0):
                dbg = self.draw_debug(frame, evs, move_act)
                cv2.imwrite(str(debug_dir / f"frame_{real_frame:06d}.jpg"), dbg)
            frames_done += 1
            if progress and frames_done % 20 == 0:
                print(f"\r  [infer] 帧 {real_frame}/{n_frames} "
                      f"动作 {len(actions)} 事件 {len(events)} "
                      f"({time.time()-t0:.0f}s)", end="", flush=True)
            read_idx += 1
        cap.release()
        if progress:
            print()

        press_events.sort(key=lambda e: e["t"])
        skill_acts = self.fuse_skills(press_events)
        actions.extend(skill_acts)
        actions.sort(key=lambda a: (a["t"] if a["t"] is not None else -1,
                                    a["type"] == "move"))
        self.stats["frames_processed"] = frames_done
        self.stats["fps"] = round(fps, 3)
        self.stats["duration_s"] = round(frames_done * sample_every / fps, 3)
        return actions, events, self.stats

    # -- 调试可视化 -----------------------------------------------------
    def draw_debug(self, frame, evs, move_act):
        vis = frame.copy()
        h, w = vis.shape[:2]
        # A
        if self.chA.enabled:
            cx, cy = int(self.chA.center[0]), int(self.chA.center[1])
            cv2.circle(vis, (cx, cy), self.chA.radius, (255, 200, 0), 1)
        for ev in evs:
            if ev["ch"] == "A":
                tip = ev.get("tip")
                if tip and self.chA.center:
                    cx, cy = int(self.chA.center[0]), int(self.chA.center[1])
                    cv2.arrowedLine(vis, (cx, cy), (int(tip[0]), int(tip[1])),
                                    (0, 255, 255), 2)
                    cv2.putText(vis, f"A:{ev['direction']} {ev['conf']:.2f}",
                                (cx, cy - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
            elif ev["ch"] == "B":
                x, y = self.chB.buttons.get(ev["button"], (0, 0))
                cv2.circle(vis, (int(x), int(y)), self.chB.r, (0, 0, 255), 2)
                cv2.putText(vis, f"B:{ev['button']}", (int(x) - 20, int(y) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif ev["ch"] == "C" and ev["kind"] == "skill_effect":
                cv2.putText(vis, f"C:skill_effect {ev['conf']:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif ev["ch"] == "D":
                cv2.putText(vis, f"D:aim {ev['skill']}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if move_act and move_act["type"] in ("move", "none"):
            col = (0, 255, 0) if move_act["type"] == "move" else (128, 128, 128)
            cv2.putText(vis, f"{move_act['type']} {move_act.get('direction','')} "
                             f"{move_act['confidence']:.2f} {move_act['channels']}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return vis


# ---------------------------------------------------------------- 主接口


def infer_actions(video_path, calib, sample_every=1, model_path=None,
                  debug_viz=None, emit_none=False, yolo_every=1,
                  max_frames=None, c_window_s=0.18, emit_aim_only=False,
                  mock_detector=None, progress=True, fps_override=None,
                  classes=None):
    """从带 HUD 录像推断玩家动作。

    参数:
      video_path  : 视频路径
      calib       : 校准 dict 或校准文件路径（calibrate_video.py 输出格式 /
                    configs/calibration.json / calibration_absolute.json）
      sample_every: 抽样帧间隔（1 = 每帧）
      model_path  : YOLO 权重路径；None 则关闭通道 C
      debug_viz   : 调试帧输出目录（None 关闭）
      emit_none   : 是否输出 type=none 帧
      yolo_every  : YOLO 运行间隔（相对采样帧计数）
      max_frames  : 最多处理的采样帧数（冒烟测试用）
      c_window_s  : C/D 证据匹配时间窗（秒）
      emit_aim_only: 是否输出仅有 D 瞄准线（无 B 按下）的拖瞄事件
      mock_detector: 测试用检测器函数（frame -> [(cls, conf, box)]）
      fps_override: 覆盖 fps
      classes     : 类别表（默认 configs/classes.json 顺序）

    返回:
      list[dict]：动作列表，末尾追加 {"type": "meta", "events": [...], "stats": {...},
      "video": ..., "sample_every": ...} 元信息条目。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    pts, res = load_calib_points(calib, video_size=(vid_w, vid_h))
    inferrer = ActionInferrer(pts, res, model_path=model_path,
                              c_window_s=c_window_s, emit_none=emit_none,
                              emit_aim_only=emit_aim_only,
                              mock_detector=mock_detector, classes=classes)
    actions, events, stats = inferrer.process_video(
        video_path, sample_every=sample_every, yolo_every=yolo_every,
        max_frames=max_frames, debug_viz=debug_viz, progress=progress,
        fps_override=fps_override)
    actions.append({"type": "meta", "t": None, "video": str(video_path),
                    "sample_every": sample_every, "events": events,
                    "stats": stats})
    return actions


def split_result(result):
    """把 infer_actions 的返回拆成 (actions, meta)。"""
    actions = [a for a in result if a.get("type") != "meta"]
    meta = next((a for a in result if a.get("type") == "meta"), {})
    return actions, meta


if __name__ == "__main__":
    import sys
    print(__doc__)
    print("请通过 scripts/train/infer_actions.py 使用命令行接口。")
    sys.exit(0)
