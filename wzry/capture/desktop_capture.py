# -*- coding: utf-8 -*-
"""桌面窗口捕获 v3: PrintWindow 平滑流(56ms/18fps) + 一次性直方图匹配色准。

色准原理: adb screencap 颜色=游戏直出(用户验证一致)。
  启动时并行采集 adb+PrintWindow 各一帧(间隔<=0.25s), 若内容一致(静止画面)
  -> 每通道直方图匹配生成 256-bin LUT -> 每帧查表。LUT 一次性=颜色恒定不漂移。
  若无静态对(全程画面在动), 用启动初期 best 对(分位)兜底; 再不行恒等。
"""
import ctypes
import threading
import time
from ctypes import wintypes

import cv2  # noqa: F401
import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class _BMIH(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


def _find_mumu():
    """v2.88 找 MuMu 窗口: ①必须可见(IsWindowVisible) ②面积最大 ③标题含'模拟器'优先。
    旧版只按面积 -> 曾选中隐藏的 MuMuNxDevice(1.79M) 而非真·MuMu模拟器12(2.03M,可见)
    -> 代理看到的永远是黑屏/空窗, 无法进对局。"""
    best = None
    best_area = 0
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def score(h, title):
        r = wintypes.RECT()
        user32.GetWindowRect(h, ctypes.byref(r))
        w, hh = r.right - r.left, r.bottom - r.top
        area = w * hh
        # v2.92 离屏窗口排除: MuMu 最小化/失焦时 rect 被移到 (-32000,-32000)
        #    IsWindowVisible 仍返回 True -> 必须检查坐标落在屏幕内
        if r.right < 0 or r.bottom < 0 or r.left > 5000 or r.top > 5000:
            return None
        if not user32.IsWindowVisible(h) or area < 40000:
            return None
        bonus = 10e9 if ("模拟器" in title) else 0   # 真主窗标题=MuMu模拟器12
        return area + bonus

    def cbk(h, l):
        nonlocal best, best_area
        n = ctypes.create_unicode_buffer(128)
        user32.GetWindowTextW(h, n, 128)
        if "MuMu" in n.value:
            s = score(h, n.value)
            if s is not None and s > best_area:
                best_area, best = s, (int(h), n.value)
        return True

    user32.EnumWindows(CB(cbk), 0)
    return best


def grab_window(hwnd):
    """PrintWindow 抓窗口 -> BGR 1280x720 (零处理, 仅保比例裁剪)。"""
    r = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    w, hh = r.right - r.left, r.bottom - r.top
    if w < 200 or hh < 200:
        return None
    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, hh)
    gdi32.SelectObject(mdc, bmp)
    ok = user32.PrintWindow(hwnd, mdc, 2)
    if not ok:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(hwnd, hdc)
        return None
    bmi = _BMIH()
    bmi.biSize = ctypes.sizeof(_BMIH)
    bmi.biWidth = w
    bmi.biHeight = -hh
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    buf = ctypes.create_string_buffer(w * hh * 4)
    gdi32.GetDIBits(mdc, bmp, 0, hh, buf, ctypes.byref(bmi), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, hdc)
    img = np.frombuffer(buf.raw, dtype=np.uint8).reshape(hh, w, 4)
    # v3.1 修正: DIB 32bpp 内存序=BGRA, 直接取前3通道即 BGR (此前误 RGBA 反转->红蓝互换色差)
    bgr = np.ascontiguousarray(img[:, :, :3])
    # 保比例中心裁剪 -> 1280x720
    if hh > 200:
        bgr = bgr[40:, :]
        h2, w2 = bgr.shape[:2]
        Wc = int(h2 * 1280 / 720)
        if Wc <= w2:
            x0 = (w2 - Wc) // 2
            bgr = bgr[:, x0:x0 + Wc]
        out = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        if out.size == 1280 * 720 * 3:
            return out
    return None


_LUT = None   # (256,3) uint8


def _build_lut(ref_bgr, src_bgr):
    """每通道直方图匹配: 让 src(PrintWindow) 分布==ref(adb)。返回 (256,3) LUT。"""
    try:
        s = cv2.resize(src_bgr, (320, 180))
        r = cv2.resize(ref_bgr, (320, 180))
        lut = np.zeros((256, 3), np.uint8)
        for c in range(3):
            sc = np.cumsum(np.bincount(s[..., c].ravel(), minlength=256)).astype(np.float64)
            rc = np.cumsum(np.bincount(r[..., c].ravel(), minlength=256)).astype(np.float64)
            sc /= sc[-1]
            rc /= rc[-1]
            lut[:, c] = np.searchsorted(rc, sc)
        return lut
    except Exception:
        return None


def _apply_lut(frame, lut):
    return lut[frame]


class DesktopCapturePrint:
    def __init__(self, interval=0.02, serial=None, rotate_s=1800):
        self.interval = interval
        self._latest = None
        self._latest_t = 0.0
        self.last_size = None
        self._stop = False
        self._thread = None
        self.last_ms = 0.0
        self._hwnd = None

    def start(self):
        self._hwnd, name = _find_mumu() or (None, None)
        print(f"桌面流: 窗口={name} hwnd={self._hwnd}")
        if self._hwnd is None:
            raise RuntimeError("未找到 MuMu 窗口")
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _calib_once(self):
        global _LUT
        try:
            from wzry.capture.screencap_capture import ScreencapCapture
            s = ScreencapCapture(interval=1.0)
            s.start()
        except Exception:
            return
        best_lut, best_sim = None, -1.0
        lin_g, lin_b = None, None
        for _ in range(30):   # 30 次尝试(每次1s), 找静止对照对或分位拟合
            time.sleep(1.0)
            try:
                ref, _ = s.wait_frame(timeout=2)
                cur = self._latest
                if ref is None or cur is None or ref.shape != cur.shape:
                    continue
                a = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
                b = cv2.resize(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
                sim = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                if sim >= 0.985:
                    lut = _build_lut(ref, cur)
                    if lut is not None:
                        best_lut, best_sim = lut, sim
                        break
                elif sim >= 0.85 and lin_g is None:
                    # 分位线性兜底(3 点)
                    dr = ref.reshape(-1, 3).astype(np.float32)
                    dc = cur.reshape(-1, 3).astype(np.float32)
                    q = (5, 50, 95)
                    rq = np.percentile(dr, q, axis=0)
                    cq = np.percentile(dc, q, axis=0)
                    g = np.stack([(rq[2] - rq[0]) / np.maximum(1e-6, cq[2] - cq[0]),
                                  (rq[1] - rq[0]) / np.maximum(1e-6, cq[1] - cq[0])]).mean(axis=0)
                    g = np.clip(g, 0.5, 2.0)
                    lin_g, lin_b = g, rq[1] - g * cq[1]
                elif sim >= 0.85:
                    pass
            except Exception:
                pass
        if best_lut is not None:
            _LUT = best_lut
            print(f"桌面流色准 LUT 已构建 (相似度 {best_sim:.3f})")
        elif lin_g is not None:
            # 生成线性 LUT
            lut = np.clip((np.arange(256)[:, None] * lin_g + lin_b), 0, 255).astype(np.uint8)
            _LUT = lut
            print(f"桌面流色准(线性兜底): gain {np.round(lin_g,3)} bias {np.round(lin_b,1)}")
        else:
            print("桌面流: 未找到可比对照, 恒等输出")
        s.stop()

    def _loop(self):
        while not self._stop:
            t0 = time.perf_counter()
            img = grab_window(self._hwnd)
            if img is not None and img.ndim == 3 and img.shape[2] == 3:
                self._latest = img
                self._latest_t = time.time()
                self.last_size = (img.shape[1], img.shape[0])
                self.last_ms = (time.perf_counter() - t0) * 1000.0
            time.sleep(self.interval)

    def wait_frame(self, timeout=5.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self._latest is not None:
                return self._latest, (time.time() - self._latest_t) * 1000.0
            time.sleep(0.02)
        return None, 0.0

    def get_frame(self):
        if self._latest is None:
            return None, 0.0
        return self._latest, (time.time() - self._latest_t) * 1000.0

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=3.0)
