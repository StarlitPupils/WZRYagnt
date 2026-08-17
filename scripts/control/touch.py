# -*- coding: utf-8 -*-
import subprocess
import time
import threading

class ADBController:
    def __init__(self, device_id="127.0.0.1:7555"):
        self.device_id = device_id
        self.holding = False
        self.hold_thread = None

    def _run(self, cmd):
        full = ['adb', '-s', self.device_id] + cmd
        print(f"执行: {' '.join(full)}")
        try:
            result = subprocess.run(full, capture_output=True, text=True, timeout=2)
            if result.returncode != 0 and result.stderr:
                print(f"ADB错误: {result.stderr.strip()}")
        except Exception as e:
            print(f"ADB异常: {e}")

    def tap(self, x, y):
        self._run(['shell', 'input', 'tap', str(x), str(y)])

    def swipe(self, x1, y1, x2, y2, duration=200):
        self._run(['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration)])

    def swipe_hold(self, x1, y1, x2, y2, duration=5000):
        """模拟按住并拖动，持续 duration 毫秒（默认5秒），期间英雄会一直朝该方向移动"""
        self.holding = True
        def hold():
            self._run(['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration)])
            self.holding = False
        self.hold_thread = threading.Thread(target=hold)
        self.hold_thread.start()

    def lift(self):
        """抬起手指（停止移动）"""
        if self.holding:
            # 发送一个空 tap 或小幅度 swipe 来中断当前长按？实际需要发送抬起事件。
            # 简单方法：使用 input keyevent 模拟触摸结束，但最可靠的是发送一个极短的 swipe 覆盖。
            # 我们采用发送一个 1ms 的 swipe 到同一位置来取消长按。
            self._run(['shell', 'input', 'swipe', '0', '0', '0', '0', '1'])
            self.holding = False
