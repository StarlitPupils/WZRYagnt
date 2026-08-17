# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

points = {}
current_label = None
frame_copy = None
frame = None

def mouse_callback(event, x, y, flags, param):
    global points, current_label, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points[current_label] = (x, y)
        print(f"{current_label}: ({x}, {y})")
        cv2.circle(frame_copy, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(frame_copy, current_label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Video Calibration', frame_copy)

def main():
    global current_label, frame_copy, frame
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
    # 跳转到10秒附近，取一帧作为参考
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 10))
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    cv2.namedWindow('Video Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Calibration', 1280, 720)
    cv2.setMouseCallback('Video Calibration', mouse_callback)

    labels = [
        'joystick_center',       # 摇杆中心（未拖动时）
        'joystick_arrow_tip',    # 摇杆方向箭头的尖端（任意方向均可）
        'skill1', 'skill2', 'skill3',
        'attack', 'recall', 'restore', 'summoner'
    ]
    print("请在画面上依次点击以下位置：")
    for label in labels:
        current_label = label
        print(f"点击 {label} ...")
        while current_label not in points:
            cv2.imshow('Video Calibration', frame_copy)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        frame_copy = frame.copy()
    cv2.destroyAllWindows()
    cap.release()
    # 保存校准
    config = {
        'video_resolution': [w, h],
        'points': points
    }
    out_path = Path(video_path).with_suffix('.calibration.json')
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"校准保存至 {out_path}")

if __name__ == "__main__":
    import sys
    main()
