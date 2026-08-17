# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path

def detect_ui_animation(video_path, sample_frames=10):
    """检测视频是否包含UI动画（摇杆/技能指示器）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 每秒取一帧
    interval = max(1, int(fps))
    motion_scores = []
    prev_gray = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # 截取摇杆区域（左下角）和技能区域（右下角）
            h, w = frame.shape[:2]
            joystick_roi = frame[int(h*0.7):h, 0:int(w*0.3)]
            skill_roi = frame[int(h*0.6):h, int(w*0.6):w]
            gray = cv2.cvtColor(cv2.vconcat([joystick_roi, skill_roi]), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(np.mean(diff))
            prev_gray = gray
        frame_idx += 1
        if len(motion_scores) >= sample_frames:
            break
    cap.release()
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    # 阈值可调整
    return avg_motion > 5.0
