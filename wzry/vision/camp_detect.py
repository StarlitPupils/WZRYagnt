# -*- coding: utf-8 -*-
"""阵营检测器 v12.4 (根治): 检测"选完英雄展示阶段"(每个人英雄卡展示, 未进游戏)
该阶段特征: 画面中部有"VS"对立标志 + 上下各5个英雄卡 + "阵营"色(蓝/红条)
此时: 看自己的金色名字(Starliit-001)在上方(卡下)还是下方 -> 上=蓝/下=红
避免误判: 仅在展示阶段(非对局, 非主页)检测。

用法:
  from wzry.vision.camp_detect import detect_show_stage, gold_name_row
"""
import cv2
import numpy as np


def detect_show_stage(frame):
    """检测选完英雄展示阶段(上下排英雄卡+VS)。返回 bool。"""
    try:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 特征1: 画面中部有 VS 大字 (白字 y300-420, x550-750)
        vs_roi = frame[300:420, 540:760]
        if vs_roi.size == 0:
            return False
        hs = cv2.cvtColor(vs_roi, cv2.COLOR_BGR2HSV)
        # VS 字白色高亮
        vs_white = int(((hs[..., 1] < 60) & (hs[..., 2] > 200)).sum())
        # 特征2: 上下两排英雄卡区 (y40-280 上, y440-660 下) 有彩色英雄头像(many colors)
        up = frame[40:280, 100:1180]
        lo = frame[440:660, 100:1180]
        up_std = float(up.std()) if up.size else 0.0
        lo_std = float(lo.std()) if lo.size else 0.0
        # 特征3: 无小地图(左上角简洁=非对局), 无比分HUD (右上无白字比分)
        mm_roi = frame[10:210, 10:230]
        mm_mean = float(mm_roi.mean()) if mm_roi.size else 0.0
        return vs_white > 20 and up_std > 30 and lo_std > 30 and mm_mean < 60
    except Exception:
        return False


def gold_name_row(frame):
    """选完英雄展示阶段: 金色名字像素 上半屏 vs 下半屏 -> 'up'/'down'/None。"""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(int)
        S = hsv[..., 1].astype(int)
        V = hsv[..., 2].astype(int)
        # 金色名字 (V>170 亮金, 低饱和文字边缘? 名字金色高亮)
        gold = ((H >= 12) & (H <= 45) & (S > 90) & (V > 150)).astype(np.uint8)
        ys = np.nonzero(gold[:, 100:1180])[0]
        if len(ys) < 25:
            return None
        up = int((ys < 360).sum())
        lo = int((ys >= 360).sum())
        # v12.7 名字金色: 在哪排卡片区域(上排卡片 y40-280 / 下排 y440-660), 取卡区内金色多者
        _up_card = int((((ys >= 40) & (ys < 280))).sum())
        _lo_card = int((((ys >= 440) & (ys < 660))).sum())
        if _up_card > 30 and _up_card > _lo_card * 1.5:
            return "up"
        if _lo_card > 30 and _lo_card > _up_card * 1.5:
            return "down"
        return None
    except Exception:
        return None
