# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import mss
import win32gui
import win32con
from pathlib import Path

def get_mumu_window_handle():
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if 'MuMu' in title:
                hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

class ScreenCapture:
    def __init__(self):
        self.hwnd = None
        self.sct = mss.mss()
        self.window_rect = None

    def find_window(self):
        self.hwnd = get_mumu_window_handle()
        return self.hwnd is not None

    def get_frame(self):
        if self.hwnd is None:
            if not self.find_window():
                return None
        rect = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
        right, bottom = win32gui.ClientToScreen(self.hwnd, (rect[2], rect[3]))
        width = right - left
        height = bottom - top
        self.window_rect = {"left": left, "top": top, "width": width, "height": height}
        try:
            screenshot = self.sct.grab(self.window_rect)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except:
            return None

points = {}
current_label = None
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global points, current_label, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points[current_label] = (x, y)
        print(f"{current_label}: ({x}, {y})")
        cv2.circle(frame_copy, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(frame_copy, current_label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Calibration', frame_copy)

def main():
    global current_label, frame_copy
    cap = ScreenCapture()
    print("等待 MuMu 窗口...")
    while not cap.find_window():
        time.sleep(1)
    frame = cap.get_frame()
    if frame is None:
        print("无法获取画面")
        return
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', 800, 600)
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

    # 保存绝对坐标和画面尺寸
    config = {
        'screen_width': w,
        'screen_height': h,
        'points': points
    }
    config_path = Path("configs/calibration_absolute.json")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"校准完成，已保存至 {config_path}")

if __name__ == "__main__":
    import time
    main()
