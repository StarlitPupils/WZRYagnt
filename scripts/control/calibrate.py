# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path
import mss
import pygetwindow as gw

def select_window():
    windows = gw.getAllWindows()
    visible_windows = [w for w in windows if w.visible and w.title.strip() != '']
    if not visible_windows:
        print("未找到任何可见窗口。")
        return None
    print("可见窗口列表：")
    for i, w in enumerate(visible_windows):
        print(f"{i}: {w.title} (大小: {w.width}x{w.height})")
    choice = input("请输入要捕获的窗口编号（直接回车自动匹配包含 'MuMu' 的窗口）: ").strip()
    if choice == '':
        for w in visible_windows:
            if 'MuMu' in w.title:
                return w
        print("未找到包含 'MuMu' 的窗口。")
        return None
    try:
        idx = int(choice)
        if 0 <= idx < len(visible_windows):
            return visible_windows[idx]
    except:
        pass
    print("输入无效。")
    return None

class ScreenCapture:
    def __init__(self, window):
        self.window = window
        self.sct = mss.mss()
        self.update_window_rect()

    def update_window_rect(self):
        if self.window.isMinimized:
            self.window.restore()
        left = self.window.left + 8
        top = self.window.top + 30
        width = self.window.width - 16
        height = self.window.height - 38
        self.window_rect = {"left": left, "top": top, "width": width, "height": height}

    def get_frame(self):
        try:
            self.update_window_rect()
            screenshot = self.sct.grab(self.window_rect)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"截图错误: {e}")
            return None

points = {}
current_label = None
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global points, current_label, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points[current_label] = (x, y)
        print(f"{current_label}: ({x}, {y})")
        cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_copy, current_label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow('Calibration', frame_copy)

def main():
    global current_label, frame_copy
    win = select_window()
    if win is None:
        return
    capture = ScreenCapture(win)
    frame = capture.get_frame()
    if frame is None:
        print("无法获取画面。")
        return
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', int(w*0.5), int(h*0.5))
    cv2.setMouseCallback('Calibration', mouse_callback)

    labels = ['move_stick_center', 'skill1', 'skill2', 'skill3', 'attack', 'minimap_center']
    print("请按顺序点击以下位置（在画面中单击左键）：")
    for label in labels:
        current_label = label
        print(f"请点击 {label} ...")
        while current_label not in points:
            cv2.imshow('Calibration', frame_copy)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        frame_copy = frame.copy()

    cv2.destroyAllWindows()

    norm_points = {k: (x/w, y/h) for k, (x, y) in points.items()}
    config_path = Path("configs/calibration.json")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(norm_points, f, indent=2)
    print(f"校准完成，已保存至 {config_path}")

if __name__ == "__main__":
    main()
