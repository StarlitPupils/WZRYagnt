# -*- coding: utf-8 -*-
"""ADB 设备帧缓冲采集：adb exec-out screencap。

优点：
  - 不依赖模拟器窗口可见/前台（窗口最小化也能采到游戏画面）；
  - 采集与 adb 输入同处设备像素坐标系，无需窗口<->设备坐标映射。
缺点：比窗口 mss 采集慢（PNG 编解码，通常 40-120ms）。
"""
import subprocess
import time

import cv2
import numpy as np

from wzry.control.executor import discover_mumu_adb_devices


class AdbCapture:
    """基于 adb screencap 的采集器。get_frame() 返回 (BGR ndarray, latency_ms)。"""

    def __init__(self, serial=None, auto_connect=True):
        self.serial = serial
        self.auto_connect = auto_connect
        self.last_latency_ms = 0.0
        self.last_size = None
        if auto_connect and self.serial is None:
            self._ensure_device()

    def _ensure_device(self):
        serials = discover_mumu_adb_devices()
        if not serials:
            raise RuntimeError("未发现 MuMu ADB 设备，请先启动模拟器")
        self.serial = serials[0]

    def get_frame(self):
        if self.serial is None:
            self._ensure_device()
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                ["adb", "-s", self.serial, "exec-out", "screencap"],
                capture_output=True, timeout=10,
            )
        except Exception:
            self.last_latency_ms = 0.0
            return None, 0.0
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        if r.returncode != 0 or not r.stdout:
            return None, self.last_latency_ms
        frame = self._decode_raw(r.stdout)
        if frame is None:
            # 兜底：PNG 模式
            try:
                r2 = subprocess.run(
                    ["adb", "-s", self.serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=10,
                )
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
                buf = np.frombuffer(r2.stdout, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                frame = None
        if frame is None:
            return None, self.last_latency_ms
        self.last_size = (frame.shape[1], frame.shape[0])
        return frame, self.last_latency_ms

    @staticmethod
    def _decode_raw(data):
        """解析裸 screencap：16 字节头(w,h,fmt,..) + RGBA8888 像素。失败返回 None。"""
        try:
            if len(data) < 16:
                return None
            w = int.from_bytes(data[0:4], "little")
            h = int.from_bytes(data[4:8], "little")
            body = data[16:]
            if not (0 < w <= 4096 and 0 < h <= 4096):
                return None
            if len(body) == w * h * 4:          # RGBA8888
                img = np.frombuffer(body, dtype=np.uint8).reshape(h, w, 4)
                return img[..., :3][:, :, ::-1].copy()   # RGBA -> BGR
            if len(body) == w * h * 3:          # RGB888
                img = np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)
                return img[:, :, ::-1].copy()
        except Exception:
            return None
        return None
