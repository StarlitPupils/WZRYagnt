# -*- coding: utf-8 -*-
"""实时采集 v2.23：adb screencap 轮询（同 ScrcpyStreamCapture 接口）。

用户要求：低延迟实时抽帧、不读历史切片。
screencap 每 0.25s 拉一张 1280x720（约 80-150ms/张），永远读当前屏幕。
"""
import subprocess
import threading
import time

import cv2
import numpy as np


class ScreencapCapture:
    def __init__(self, interval=0.02, serial=None, rotate_s=1800):
        self.interval = interval
        self.serial = serial or "127.0.0.1:16384"
        self.rotate_s = rotate_s
        self._latest = None
        self._latest_t = 0.0
        self.last_size = None
        self._stop = False
        self._thread = None
        self._session_start = 0.0
        self.last_ms = 0.0
        self._fail_streak = 0

    def start(self):
        import subprocess as sp
        # 确保 adb 连接
        sp.run(["adb", "connect", self.serial], capture_output=True, timeout=15)
        self._stop = False
        self._session_start = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _adb(self, args, timeout=3.0):
        """执行 adb 命令, 带断线自愈: 失败自动 reconnect 重试。"""
        for attempt in range(3):
            try:
                p = subprocess.run(
                    ["adb", "-s", self.serial] + args,
                    capture_output=True, timeout=timeout)
                if p.returncode == 0 and p.stdout:
                    return p.stdout
            except Exception:
                pass
            # 自愈: 重连
            try:
                subprocess.run(["adb", "connect", self.serial],
                               capture_output=True, timeout=10)
            except Exception:
                pass
            time.sleep(0.4 * (attempt + 1))
        return None

    def _grab(self):
        """RAW screencap(免PNG编码, ~175ms) -> BGR 帧。测试证明 640 抓取无增益(adb固定开销)。"""
        t0 = time.perf_counter()
        raw = self._adb(["exec-out", "screencap"])
        if raw is None or len(raw) < 4 * 1280 * 720 + 16:
            return None
        buf = np.frombuffer(raw[16:16 + 1280 * 720 * 4], dtype=np.uint8)
        img_rgba = buf.reshape(720, 1280, 4)
        img = np.ascontiguousarray(img_rgba[:, :, :3][:, :, ::-1])  # RGBA->BGR
        # v2.38 RAW 帧偏暗 ~10% -> 增益对齐 PNG 标定色域
        img = np.clip(img.astype(np.float32) * 1.15 + 4.0, 0, 255).astype(np.uint8)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return img

    def _loop(self):
        while not self._stop:
            img = self._grab()
            if img is not None:
                self._fail_streak = 0
                self._latest = img
                self._latest_t = time.time()
                self.last_size = (img.shape[1], img.shape[0])
            else:
                self._fail_streak += 1
            time.sleep(self.interval)

    def wait_frame(self, timeout=5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self._latest is not None and time.time() - self._latest_t < 1.0:
                return self._latest, (time.time() - self._latest_t) * 1000.0
            time.sleep(0.03)
        return None, 0.0

    def get_frame(self):
        if self._latest is None:
            return None, 0.0
        return self._latest, (time.time() - self._latest_t) * 1000.0

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=3.0)
