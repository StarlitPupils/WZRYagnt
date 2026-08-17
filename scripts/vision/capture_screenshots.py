# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import mss
import win32gui
import win32con
from pathlib import Path
from datetime import datetime

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
        except Exception as e:
            print(f"截图错误: {e}")
            return None

class ScreenshotCollector:
    def __init__(self, hero_name="zhongkui", save_dir="data/screenshots"):
        self.hero_name = hero_name
        self.save_dir = Path(save_dir) / hero_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.capture = ScreenCapture()

    def collect(self, interval=1.0, max_count=None):
        print("等待 MuMu 模拟器窗口...")
        while not self.capture.find_window():
            time.sleep(1)
        print(f"开始采集，间隔 {interval} 秒，保存至 {self.save_dir}")
        print("按 Ctrl+C 停止")
        count = 0
        try:
            while True:
                frame = self.capture.get_frame()
                if frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.hero_name}_{timestamp}_{count:06d}.jpg"
                    filepath = self.save_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    count += 1
                    print(f"[{count}] 已保存: {filename}")
                else:
                    print("无法获取画面，重试...")
                if max_count and count >= max_count:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            print(f"采集结束，共保存 {count} 张截图")

if __name__ == "__main__":
    collector = ScreenshotCollector(hero_name="zhongkui")
    collector.collect(interval=1.0)
