# -*- coding: utf-8 -*-
"""统一窗口采集：mss + win32gui 抓取 MuMu 模拟器窗口。

M0 版本：单实例，按窗口标题关键字匹配（默认 'MuMu'）。
输出 BGR ndarray（与 cv2 一致），并记录每次采集耗时。
"""
import time

import numpy as np
import mss
import win32gui


def find_mumu_window(title_keyword="MuMu"):
    """返回第一个可见且标题含关键字的窗口句柄；找不到返回 None。"""
    hwnds = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_keyword in title:
                hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    return hwnds[0] if hwnds else None


class WindowCapture:
    """窗口截图器：get_frame() 返回 (frame, latency_ms)；失败返回 (None, 0)。"""

    def __init__(self, title_keyword="MuMu"):
        self.title_keyword = title_keyword
        self.hwnd = None
        self.sct = mss.mss()
        self.last_latency_ms = 0.0

    def find_window(self):
        self.hwnd = find_mumu_window(self.title_keyword)
        return self.hwnd is not None

    def get_frame(self):
        t0 = time.perf_counter()
        if self.hwnd is None or not win32gui.IsWindow(self.hwnd):
            if not self.find_window():
                self.last_latency_ms = 0.0
                return None, 0.0
        rect = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
        right, bottom = win32gui.ClientToScreen(self.hwnd, (rect[2], rect[3]))
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            self.last_latency_ms = 0.0
            return None, 0.0
        try:
            shot = self.sct.grab({"left": left, "top": top, "width": width, "height": height})
            frame = np.asarray(shot)[..., :3]  # BGRA -> BGR
        except Exception:
            self.last_latency_ms = 0.0
            return None, 0.0
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return frame, self.last_latency_ms

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass
