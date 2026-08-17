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
    """返回第一个可见且标题含关键字的窗口句柄；找不到返回 None。

    注意：多实例/多窗口时标题匹配不可靠，优先用 mumu_render_hwnd()。
    """
    hwnds = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_keyword in title:
                hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    return hwnds[0] if hwnds else None


def find_largest_mumu_window(title_keyword="MuMu"):
    """返回可见且标题含关键字、客户区面积最大的窗口（多窗口时的兜底选择）。"""
    best = None
    best_area = -1

    def callback(hwnd, _):
        nonlocal best, best_area
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title_keyword not in title:
            return True
        try:
            r = win32gui.GetClientRect(hwnd)
            area = r[2] * r[3]
            if area > best_area:
                best, best_area = hwnd, area
        except Exception:
            pass
        return True

    win32gui.EnumWindows(callback, None)
    return best


def mumu_render_hwnd(manager_paths=None, vm_index=0):
    """从 MuMuManager 查询指定实例的游戏渲染窗口句柄（render_wnd）。

    返回 hwnd 或 None。manager_paths 默认探测常见 MuMu12 安装路径。
    """
    import json
    import subprocess
    from pathlib import Path

    defaults = [
        r"E:\MuMuPlayer\nx_main\MuMuManager.exe",
        r"C:\Program Files\Netease\MuMu Player 12\shell\MuMuManager.exe",
        r"D:\Program Files\Netease\MuMu Player 12\shell\MuMuManager.exe",
        r"C:\Program Files\Netease\MuMu\nx_main\MuMuManager.exe",
        r"D:\Program Files\Netease\MuMu\nx_main\MuMuManager.exe",
    ]
    for p in (manager_paths or defaults):
        if not Path(p).exists():
            continue
        try:
            r = subprocess.run([p, "info", "-v", "all"],
                               capture_output=True, timeout=10)
            out = r.stdout.decode("utf-8", errors="replace")
            data = json.loads(out)
            vm = data.get(str(vm_index), {})
            hw = vm.get("render_wnd")
            if hw:
                return int(hw, 16)
        except Exception:
            continue
    return None


class WindowCapture:
    """窗口截图器：get_frame() 返回 (frame, latency_ms)；失败返回 (None, 0)。

    窗口选择优先级：显式 hwnd > MuMuManager render_wnd（游戏渲染窗口）> 最大 MuMu 标题窗口。
    """

    def __init__(self, title_keyword="MuMu", hwnd=None, prefer_render=True, vm_index=0):
        self.title_keyword = title_keyword
        self.prefer_render = prefer_render
        self.vm_index = vm_index
        self.sct = mss.mss()
        self.last_latency_ms = 0.0
        self.hwnd = None
        if hwnd:
            self.hwnd = hwnd
        elif prefer_render:
            self.hwnd = mumu_render_hwnd(vm_index=vm_index) or None

    def find_window(self):
        if self.hwnd and win32gui.IsWindow(self.hwnd):
            return self.hwnd
        if self.prefer_render:
            hw = mumu_render_hwnd(vm_index=self.vm_index)
            if hw and win32gui.IsWindow(hw):
                self.hwnd = hw
                return hw
        self.hwnd = find_largest_mumu_window(self.title_keyword)
        return self.hwnd

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
