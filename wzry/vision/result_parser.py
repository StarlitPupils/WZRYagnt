# -*- coding: utf-8 -*-
"""结算面板解析器 (v12.8): 对局结束结算页读取 Starliit 行 KDA/伤害/承伤/经济。
结算页特征: 顶部"胜利/失败"金色大字 + 左右阵营列表 (我方左/敌右, 我方蓝/红方).
检测 Starliit-001 文字行 -> 读该行数值 (击杀/死亡/助攻/经济/伤害占比等难OCR,
简化为: 用 yolo/文字检测读 KDA; 伤害/承伤从结算页下半区数值栏).

简化可靠版: 结算页 KDA 我用模板/行特征 (我方列表第一行 = Starliit) 直接读:
  分辨率1280x720 结算页, Starliit 行 y≈250-280, 数值列 x 440-620 (击杀/死亡/助攻/经济).
返回 {"kill":..,"died":..,"assist":..} 或 None.
"""
import re

import cv2
import numpy as np


def detect_result_screen(frame):
    """检测结算页(顶部胜利/失败大字)。"""
    try:
        roi = frame[10:60, 400:880]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 金色/红色大字 (胜利金/失败红)
        gold = int(((hsv[..., 0] >= 10) & (hsv[..., 0] <= 45) & (hsv[..., 2] > 180)).sum())
        return gold > 200
    except Exception:
        return False


def parse_kda(frame):
    """读结算页 Starliit 行 KDA。简化: 我方列表第一行区域扫描数字列。"""
    try:
        # 我方列表 (左半 x 80-620, Starliit 行通常在 y 200-300)
        # 数染色: 找 Starliit 名牌 (白字), 该行附近数字.
        h, w = frame.shape[:2]
        # 英雄名 "Starliit-001" 白色行 (y 230-270, x 180-320)
        # 读取该行右侧数值: 击杀x 死亡x 助攻x (白/蓝数字, x 440-600)
        # OCR 无 -> 用数字颜色块检测: 击杀=白数字, 死亡=红/白
        return None   # 简化为 None (后续接 OCR)
    except Exception:
        return None


def parse_damage(frame):
    """伤害/承伤占比: 结算页下半区有"伤害占比/承伤占比"文字+数值(难OCR, 预留)."""
    return None
