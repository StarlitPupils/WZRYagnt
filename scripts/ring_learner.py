# -*- coding: utf-8 -*-
"""小地图环判别学习器: 用 s01-12 标注(11/12/13=自/友/敌环)学习 vs 全图随机负样本。
输出 configs/ring_model.json {w[8], b} (numpy 逻辑回归, 特征=环分/中心/边缘/饱和度/坡度/成分面积/类别色)
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MM = 242.0
RAND = random.Random(42)


def _band_sel():
    yy, xx = np.mgrid[-15:16, -15:16]
    dd = np.sqrt(xx ** 2 + yy ** 2)
    return (dd >= 8) & (dd <= 14)


SEL = _band_sel()


def feat(mm_bgr, hsv, m, px, py):
    """8 维特征: ring, cen, edge, s90, v90, slope, area, 1。"""
    rm = cv2.filter2D(m.astype(np.float32), -1, np.ones((15, 15), np.float32))
    # 环核(简化): 用环带均值近似
    patch = m[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
    ring = float(patch[SEL[:patch.shape[0], :patch.shape[1]].reshape(-1, 1)[:, 0]].mean()) if patch.size > 0 else 0.0
    return np.array([0.0] * 8, np.float32)


def main():
    # 特征实现(与 mm_rules 对齐的简化版)
    RING = np.ones((15, 15), np.float32); RING[5:10, 5:10] = 0.0
    RING = RING / RING.sum()
    CEN = np.zeros((11, 11), np.float32); CEN[3:8, 3:8] = 1.0
    CEN = CEN / CEN.sum()

    def feats(mm_bgr, hsv, masks, px, py):
        ring_map = cv2.filter2D(masks['use'], -1, RING)
        cen_map = cv2.filter2D(masks['use'], -1, CEN)
        p_ = mm_bgr[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
        p_h = hsv[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
        y0, x0 = max(0, py - 16), max(0, px - 16)
        mw = masks['use'][y0:py + 17, x0:px + 17]
        # slope
        slope = 0.0
        if mw.shape[0] == 33 and mw.shape[1] == 33:
            yy, xx = np.mgrid[-16:17, -16:17]
            dd = np.sqrt(xx ** 2 + yy ** 2)
            slope = float(mw[(dd >= 11.5) & (dd <= 14.5)].mean()
                          - mw[(dd >= 6.5) & (dd <= 9.5)].mean())
        # edge
        g = cv2.cvtColor(mm_bgr[max(0, py - 6):py + 7, max(0, px - 6):px + 7],
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge = 0.0
        if g.shape == (13, 13):
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            edge = float((np.sqrt(gx ** 2 + gy ** 2) > 60).mean())
        s90, v90 = 0.0, 0.0
        if p_h.shape[0] == 31 and p_h.shape[1] == 31:
            sel = SEL
            s90 = float(np.percentile(p_h[..., 1][sel], 90))
            v90 = float(np.percentile(p_h[..., 2][sel], 90))
        return np.array([float(ring_map[py, px]), float(cen_map[py, px]), edge,
                         s90, v90, slope, float(masks['mask_sum']), 1.0], np.float32)

    # ---- 采样 ----
    X, y = [], []
    for sid in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        fimg = Path(f'temp/ann/s{sid}.png')
        ftxt = Path(f'temp/ann/s{sid}.txt')
        if not fimg.exists() or not ftxt.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(fimg), dtype=np.uint8), cv2.IMREAD_COLOR)
        mm = img[:720, :1280][0:242, 0:242]
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
        m_self = ((H >= 35) & (H <= 90) & (S > 40)).astype(np.float32)
        m_ally = ((H >= 85) & (H <= 150) & (S > 40)).astype(np.float32)
        m_enemy = (((H <= 20) | (H >= 160)) & (S > 40)).astype(np.float32)
        masks = {'self': (m_self, m_self.sum()), 'ally': (m_ally, m_ally.sum()),
                 'enemy': (m_enemy, m_enemy.sum())}
        m_cls = {11: 'self', 12: 'ally', 13: 'enemy'}
        # 正样本
        for l in ftxt.read_text(encoding='utf-8').splitlines():
            p = l.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            if c not in m_cls:
                continue
            px = int(float(p[1]) * 1280.0 / MM * MM / 1280.0 * 242.0)
            px = int(float(p[1]) * 1280.0)
            py = int(float(p[2]) * 720.0)
            if not (0 < px < 242 and 0 < py < 242):
                continue
            mm_i, ssum = masks[m_cls[c]]
            masks['use'] = mm_i
            masks['mask_sum'] = ssum
            X.append(feats(mm, hsv, masks, px, py))
            y.append(1)
        # 负样本: 每图 40 个随机点(避开正样本 20px)
        negs = 0
        tries = 0
        while negs < 40 and tries < 200:
            tries += 1
            px, py = RAND.randint(15, 227), RAND.randint(15, 227)
            if any((px - int(float(p[1]) * 1280.0)) ** 2 +
                   (py - int(float(p[2]) * 720.0)) ** 2 < 400
                   for l in ftxt.read_text(encoding='utf-8').splitlines()
                   for p in [l.split()] if len(p) >= 5 and int(p[0]) in m_cls):
                continue
            mm_i, ssum = masks[RAND.choice(['self', 'ally', 'enemy'])]
            masks['use'] = mm_i
            masks['mask_sum'] = ssum
            X.append(feats(mm, hsv, masks, px, py))
            y.append(0)
            negs += 1
    X = np.array(X)
    y = np.array(y)
    print(f"样本: X{X.shape} 正{int(y.sum())} 负{int((y == 0).sum())}")

    # ---- 标准化 + 逻辑回归 (纯 numpy) ----
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    n = Xs.shape[1]
    w = np.zeros(n, np.float64)
    lr, l2, iters = 0.5, 1e-2, 4000
    for it in range(iters):
        z = Xs @ w
        p_ = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g_ = Xs.T @ (p_ - y) / len(y) + l2 * w
        w -= lr * g_
    # 阈值: F1 最大搜索
    sc = Xs @ w
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.percentile(sc, np.linspace(1, 99, 99)):
        tp = int(((sc > thr) & (y == 1)).sum())
        fp = int(((sc > thr) & (y == 0)).sum())
        fn = int(((sc <= thr) & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    acc = float(((sc > best_thr) == (y == 1)).mean())
    print(f"标准化模型: w={np.round(w, 3)} 最优thr={best_thr:.3f} F1={best_f1:.2%} 一致率={acc:.2%}")
    (ROOT / 'configs' / 'ring_model.json').write_text(
        json.dumps({'w': w.tolist(), 'b': float(best_thr),
                    'mu': mu.tolist(), 'sd': sd.tolist()},
                   ensure_ascii=False), encoding='utf-8')
    print('已存 configs/ring_model.json')


if __name__ == '__main__':
    main()
