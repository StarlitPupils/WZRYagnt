# -*- coding: utf-8 -*-
"""Minimap rules det v7 (rewritten, ASCII comments only).

Detects hero rings (self/ally/enemy), minion dots (template match),
fixed towers/monsters/buff points on the minimap. Key params were tuned
for: adb RAW frames (S 60+), PrintWindow frames (S 40+), and corner
decorations (bottom-right blue block / crystal / triangle).

Detector output keys: found/center/radius/size, dots{self,ally,enemy,monster},
minions{ally,enemy}, towers{ally,enemy}, buff.
"""
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MM_SIZE = 242
_RING = np.ones((15, 15), np.float32)
_RING[5:10, 5:10] = 0.0
_RING = _RING / _RING.sum()
_CENTER = np.zeros((11, 11), np.float32)
_CENTER[3:8, 3:8] = 1.0
_CENTER = _CENTER / _CENTER.sum()
_BAND_SEL = None


def _band_sel():
    global _BAND_SEL
    if _BAND_SEL is None:
        yy, xx = np.mgrid[-15:16, -15:16]
        dd = np.sqrt(xx ** 2 + yy ** 2)
        _BAND_SEL = (dd >= 8) & (dd <= 14)
    return _BAND_SEL


RING_RATIO = {"self": 0.30, "ally": 0.35, "enemy": 0.35}
CENTER_CAP = {"self": 0.15, "ally": 0.15, "enemy": 0.15}

# fixed points (normalized 0-1 of 242px; rebuilt from s01-s12 GT cluster)
BLUE_TOWER_PTS = [(0.032, 0.300), (0.031, 0.535), (0.045, 0.717), (0.082, 0.847),
                  (0.169, 0.783), (0.222, 0.907), (0.450, 0.906), (0.301, 0.661),
                  (0.366, 0.545), (0.695, 0.906)]
RED_TOWER_PTS = [(0.293, 0.051), (0.500, 0.042), (0.711, 0.042), (0.779, 0.168),
                 (0.869, 0.092), (0.653, 0.277), (0.905, 0.223), (0.916, 0.415),
                 (0.907, 0.701), (0.578, 0.390)]
MONSTER_PTS = [(0.590, 0.178), (0.319, 0.130), (0.500, 0.307), (0.171, 0.284),
               (0.187, 0.549), (0.355, 0.742), (0.449, 0.631), (0.625, 0.801),
               (0.778, 0.656), (0.803, 0.508), (0.746, 0.388), (0.065, 0.051),
               (0.930, 0.917), (0.141, 0.432)]
BUFF_PTS = [(0.685, 0.499), (0.504, 0.743), (0.444, 0.198), (0.270, 0.440),
            (0.313, 0.282), (0.626, 0.676)]

_RING_MODEL = None


def _load_ring_model():
    global _RING_MODEL
    if _RING_MODEL is None:
        try:
            import json
            p = ROOT / "configs" / "ring_model.json"
            if p.exists():
                _RING_MODEL = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _RING_MODEL = False
    return _RING_MODEL


def ring_model_score(mm_bgr, hsv, m, px, py):
    """v2.63 学到的环判别分(8维, z-score 逻辑回归)。"""
    md = _load_ring_model()
    if not md:
        return None
    f = ring_feat(mm_bgr, hsv, m, px, py)
    xs = (f - np.array(md["mu"])) / np.array(md["sd"])
    return float(xs @ np.array(md["w"]))


def ring_feat(mm_bgr, hsv, m, px, py):
    RINGK = np.ones((15, 15), np.float32)
    RINGK[5:10, 5:10] = 0.0
    RINGK = RINGK / RINGK.sum()
    CENK = np.zeros((11, 11), np.float32)
    CENK[3:8, 3:8] = 1.0
    CENK = CENK / CENK.sum()
    ring_map = cv2.filter2D(m, -1, RINGK)
    cen_map = cv2.filter2D(m, -1, CENK)
    patch_h = hsv[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
    s90, v90 = 0.0, 0.0
    if patch_h.shape[0] == 31 and patch_h.shape[1] == 31:
        sel = _band_sel()
        s90 = float(np.percentile(patch_h[..., 1][sel], 90))
        v90 = float(np.percentile(patch_h[..., 2][sel], 90))
    g = cv2.cvtColor(mm_bgr[max(0, py - 6):py + 7, max(0, px - 6):px + 7],
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge = 0.0
    if g.shape == (13, 13):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edge = float((np.sqrt(gx ** 2 + gy ** 2) > 60).mean())
    mw = m[max(0, py - 16):py + 17, max(0, px - 16):px + 17]
    slope = 0.0
    if mw.shape[0] == 33 and mw.shape[1] == 33:
        yy, xx = np.mgrid[-16:17, -16:17]
        dd = np.sqrt(xx ** 2 + yy ** 2)
        slope = float(mw[(dd >= 11.5) & (dd <= 14.5)].mean()
                      - mw[(dd >= 6.5) & (dd <= 9.5)].mean())
    return np.array([float(ring_map[py, px]), float(cen_map[py, px]), edge,
                     s90, v90, slope, float(m.sum()), 1.0], np.float32)


def _edge_frac(mm_bgr, cx, cy, half=6):
    patch = mm_bgr[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
    if patch.shape[0] != 2 * half + 1 or patch.shape[1] != 2 * half + 1:
        if patch.size == 0:
            return None
        patch = cv2.copyMakeBorder(patch, 0, 2 * half + 1 - patch.shape[0],
                                   0, 2 * half + 1 - patch.shape[1],
                                   cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float((mag > 60).mean())


def _has_avatar(mm_bgr, cx, cy, half=6, thr=0.75):
    ed = _edge_frac(mm_bgr, cx, cy, half)
    return ed is not None and ed > thr


def _ring_slope(m, cx, cy):
    win = m[max(0, cy - 16):cy + 17, max(0, cx - 16):cx + 17]
    if win.shape[0] != 33 or win.shape[1] != 33:
        return -1.0
    yy, xx = np.mgrid[-16:17, -16:17]
    dd = np.sqrt(xx ** 2 + yy ** 2)
    f8 = float(win[(dd >= 6.5) & (dd <= 9.5)].mean())
    f13 = float(win[(dd >= 11.5) & (dd <= 14.5)].mean())
    return f13 - f8


def _tower_alive(mm_bgr, cx, cy, side, search=7, min_px=3):
    ya, yb = max(0, cy - search), min(mm_bgr.shape[0], cy + search + 1)
    xa, xb = max(0, cx - search), min(mm_bgr.shape[1], cx + search + 1)
    patch = mm_bgr[ya:yb, xa:xb]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    if side == "ally":
        m = ((H >= 85) & (H <= 135) & (S > 50)).astype(np.float32)
    else:
        m = ((((H <= 20) | (H >= 160)) & (S > 50))).astype(np.float32)
    conv = cv2.filter2D(m, -1, np.ones((5, 5), np.float32))
    return float(conv.max()) >= min_px


def _ring_centroid(ring_map, px, py, win=6, thr=0.85):
    y0, x0 = max(0, py - win), max(0, px - win)
    wm = ring_map[y0:py + win + 1, x0:px + win + 1]
    if wm.size == 0:
        return float(px), float(py)
    m = wm >= thr * max(1e-6, float(ring_map[py, px]))
    if not m.any():
        return float(px), float(py)
    ys, xs = np.nonzero(m)
    return int(round(x0 + float(xs.mean()))), int(round(y0 + float(ys.mean())))


def _prune_templates(arr, max_t):
    """k-means-lite clustering by position; keep best-scored per cluster."""
    if len(arr) <= max_t:
        return arr
    out = []
    for t in arr:
        merged = False
        for o in out:
            if abs(o[0] - t[0]) < 8 and abs(o[1] - t[1]) < 8:
                if t[2] > o[2]:
                    o[0], o[1], o[2] = t[0], t[1], t[2]
                merged = True
                break
        if not merged:
            out.append([t[0], t[1], t[2], t[3]])
    return out[:max_t]


_DOT_TEMPLATES = {"ally": None, "enemy": None}
_DOT_THR = 0.70
_DOT_MAX_T = 18


def _prune_templates_legacy(arr, max_t):
    return _prune_templates(arr, max_t)


def _dot_match(mm, templates, thr=0.62, pad=8):
    res = []
    g = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for (tx, ty, score, tpl_path) in templates:
        tpl = np.load(tpl_path) if isinstance(tpl_path, str) else tpl_path
        if tpl.ndim == 3:
            tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        tpl = tpl.astype(np.float32)
        th, tw = tpl.shape[:2]
        if mm.shape[0] < th or mm.shape[1] < tw:
            continue
        rc = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
        while True:
            _, mv, _, mloc = cv2.minMaxLoc(rc)
            if mv < thr:
                break
            bx, by = mloc
            cx, cy = bx + tw // 2, by + th // 2
            res.append((cx, cy))
            rc[max(0, by - 3):by + th + 3, max(0, bx - 3):bx + tw + 3] = -1
    # NMS 6px
    res = _nms(res, 6)
    return res


def _nms(pts, d=6):
    out = []
    for p in sorted(pts, key=lambda q: -q[0]):
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > d for q in out):
            out.append(p)
    return out


def _load_dot_templates():
    global _DOT_TEMPLATES
    if _DOT_TEMPLATES["ally"] is not None:
        return
    base = ROOT / "temp" / "mm_dot_templates"
    ally = []
    enemy = []
    for f in sorted(base.glob("ally_*.npy")):
        tpl = np.load(str(f))
        ally.append((0, 0, 0.9, tpl))
    for f in sorted(base.glob("enemy_*.npy")):
        tpl = np.load(str(f))
        enemy.append((0, 0, 0.9, tpl))
    _DOT_TEMPLATES["ally"] = _prune_templates(ally, _DOT_MAX_T)
    _DOT_TEMPLATES["enemy"] = _prune_templates(enemy, _DOT_MAX_T)


class MMDetectorV7:
    def detect(self, frame, mm_box=None):
        h, w = frame.shape[:2]
        if mm_box is None:
            x0, y0 = 0, 0
            mw = mh = MM_SIZE
        else:
            x0, y0, x1, y1 = mm_box
            mw, mh = x1 - x0, y1 - y0
        mm = frame[y0:y0 + mh, x0:x0 + mw]
        if mm.size == 0 or mm.shape[0] < 30 or mm.shape[1] < 30:
            return self._empty()
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

        # ring masks (S>40 for PrintWindow; H windows tuned)
        color_masks = {
            "self": ((H >= 35) & (H <= 90) & (S > 40)).astype(np.float32),
            "ally": ((H >= 85) & (H <= 150) & (S > 40)).astype(np.float32),
            "enemy": (((H <= 20) | (H >= 160)) & (S > 40)).astype(np.float32),
        }
        res = {}
        for name, m in color_masks.items():
            ring_map = cv2.filter2D(m, -1, _RING)
            cen_map = cv2.filter2D(m, -1, _CENTER)
            out_pts = []
            for ptag, thr_ring, cen_ok in (("A", RING_RATIO[name], True),
                                           ("B", 0.62, False)):
                if ptag == "A":
                    hits = ((ring_map > RING_RATIO[name]) & (cen_map < CENTER_CAP[name])).astype(np.uint8) * 255
                else:
                    hits = (ring_map > 0.62).astype(np.uint8) * 255
                n, lab, st, cent = cv2.connectedComponentsWithStats(hits, 8)
                for i in range(1, n):
                    cx, cy = int(cent[i][0]), int(cent[i][1])
                    if not (5 <= cx <= mw - 5 and 5 <= cy <= mh - 5):
                        continue
                    win = ring_map[max(0, cy - 6):cy + 7, max(0, cx - 6):cx + 7] \
                        - cen_map[max(0, cy - 6):cy + 7, max(0, cx - 6):cx + 7] * 2.0
                    if win.size == 0 or win.shape[0] < 3 or win.shape[1] < 3:
                        continue
                    dy, dx = np.unravel_index(np.argmax(win), win.shape)
                    px, py = max(0, cx - 6) + dx, max(0, cy - 6) + dy
                    if ring_map[py, px] <= thr_ring:
                        continue
                    if cen_ok and cen_map[py, px] >= CENTER_CAP[name]:
                        continue
                    if ptag == "B":
                        px, py = _ring_centroid(ring_map, px, py)   # B 通道质心精化
                    # v2.63 学到的环判别: 模型分过低 -> 剔除(替代启发式倾向)
                    _mscore = ring_model_score(mm, hsv, m, px, py)
                    _hi = (_mscore is not None and
                           _mscore >= float(_RING_MODEL["b"]))
                    if name == "enemy":
                        # v2.77 四道硬闸防误检(野怪/红塔/场景红->不是敌英雄)
                        if not _has_avatar(mm, px, py, thr=0.62):
                            continue
                        patch = mm[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
                        if patch.shape[0] == 7 and patch.shape[1] == 7:
                            hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                            hh_, ss_ = hsv_p[..., 0].astype(int), hsv_p[..., 1].astype(int)
                            pure_red = (((hh_ <= 15) | (hh_ >= 170)) & (ss_ > 90)).sum()
                            if pure_red / 49.0 >= 0.55:
                                continue
                        # 固定塔点/野怪点排除(这些是塔图标/野怪, 非敌英雄):
                        near_bad = False
                        for (fx_, fy_) in BLUE_TOWER_PTS + RED_TOWER_PTS + MONSTER_PTS:
                            if math.hypot(px - fx_ * mw, py - fy_ * mh) < 13:
                                near_bad = True
                                break
                        if near_bad:
                            continue
                    if name in ("self", "ally") and px > 216 and py > 216:
                        continue   # corner decor (blue block/crystal/triangle)
                    if name == "self":
                        patch = hsv[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
                        pv_ok = False
                        if patch.shape[0] == 31 and patch.shape[1] == 31:
                            sel = _band_sel()
                            s90 = np.percentile(patch[..., 1][sel], 90)
                            v90 = np.percentile(patch[..., 2][sel], 90)
                            pv_ok = (s90 >= 145 and v90 >= 145)
                        elif patch.size > 0:
                            pad = cv2.copyMakeBorder(patch, 0, 31 - patch.shape[0],
                                                     0, 31 - patch.shape[1],
                                                     cv2.BORDER_REPLICATE)
                            s90 = np.percentile(pad[..., 1][_band_sel()], 90)
                            v90 = np.percentile(pad[..., 2][_band_sel()], 90)
                            pv_ok = (s90 >= 145 and v90 >= 145)
                        if not _hi and not pv_ok and ring_map[py, px] <= 0.70:
                            continue
                        if not _hi and not _has_avatar(mm, px, py, thr=0.55):
                            continue
                    elif name == "ally":
                        patch2 = hsv[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
                        if patch2.shape[0] != 31 or patch2.shape[1] != 31:
                            continue
                        s90 = np.percentile(patch2[..., 1][_band_sel()], 90)
                        avail = _edge_frac(mm, px, py)
                        if avail is None or avail <= 0.62 or s90 <= 140:
                            continue
                        if _ring_slope(m, px, py) < 0.05:
                            continue
                    if any(math.hypot(px - op[0], py - op[1]) < 8 for op in out_pts):
                        continue   # 两通道重复候选
                    out_pts.append((px, py, float(ring_map[py, px])))
            # v2.77 红点数量帽: 5v5 至多 5 个(防洪水误检)
            if name == "enemy":
                out_pts = out_pts[:5]
            res[name] = out_pts

        # v2.78 红环互斥: 与队友/自己环同点(<9px)的红点=队友图标红色纹理 -> 删
        if res.get("enemy"):
            _ally_pts = [p for pp in (res.get("ally", []) + res.get("self", []))
                         for p in (pp[:2],)]
            if _ally_pts:
                kp = []
                for e in res["enemy"]:
                    if any(math.hypot(e[0] - a[0], e[1] - a[1]) < 9 for a in _ally_pts):
                        continue
                    kp.append(e)
                res["enemy"] = kp

        # minion dots (template matching)
        _load_dot_templates()
        minions_ally = _dot_match(mm, _DOT_TEMPLATES["ally"] or [], thr=_DOT_THR)
        minions_enemy = _dot_match(mm, _DOT_TEMPLATES["enemy"] or [], thr=_DOT_THR)

        # fixed points
        towers_ally, towers_enemy = [], []
        for x, y in BLUE_TOWER_PTS:
            px, py = int(x * mw), int(y * mh)
            if _tower_alive(mm, px, py, "ally"):
                towers_ally.append((px, py))
        for x, y in RED_TOWER_PTS:
            px, py = int(x * mw), int(y * mh)
            if _tower_alive(mm, px, py, "enemy"):
                towers_enemy.append((px, py))
        monster = [(int(x * mw), int(y * mh)) for x, y in MONSTER_PTS]
        buff = [(int(x * mw), int(y * mh)) for x, y in BUFF_PTS]
        # monsters with visible yellow dot kept only (simplify: all fixed)
        in_match_evidence = bool(res["self"] or res["ally"] or res["enemy"]
                                 or minions_ally or minions_enemy)
        self_r = res["self"][:2]
        ally_r = res["ally"][:6]
        enemy_r = res["enemy"][:6]

        def norm(pts):
            return [{"n": [round(p[0] / mw, 4), round(p[1] / mh, 4)],
                     "conf": round(float(p[2]), 2) if len(p) > 2 else 0.7,
                     "src": "v7rw"} for p in pts]

        return {"found": in_match_evidence,
                "center": ((x0 + mw / 2, y0 + mh / 2)), "radius": mw / 2,
                "size": mw,
                "dots": {"self": norm(self_r), "ally": norm(ally_r),
                         "enemy": norm(enemy_r),
                         "monster": [{"n": [round(p[0] / mw, 4), round(p[1] / mh, 4)],
                                      "conf": 0.8, "src": "fixed"} for p in monster]},
                "minions": {"ally": norm(minions_ally), "enemy": norm(minions_enemy)},
                "towers": {"ally": norm(towers_ally), "enemy": norm(towers_enemy)},
                "buff": [{"n": [round(p[0] / mw, 4), round(p[1] / mh, 4)],
                          "conf": 0.8, "src": "fixed"} for p in buff]}

    def _empty(self):
        return {"found": False, "center": None, "radius": None, "size": MM_SIZE,
                "dots": {"self": [], "ally": [], "enemy": [], "monster": []},
                "minions": {"ally": [], "enemy": []},
                "towers": {"ally": [], "enemy": []}, "buff": []}
