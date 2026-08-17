# -*- coding: utf-8 -*-
import subprocess
import time

device_id = "d1cbbc0f"
phone_w, phone_h = 1440, 3200

def adb_tap(x, y):
    cmd = ['adb', '-s', device_id, 'shell', 'input', 'tap', str(x), str(y)]
    print(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd)

# 测试屏幕中心点击
center_x, center_y = phone_w // 2, phone_h // 2
print(f"将在3秒后点击手机屏幕中心 ({center_x}, {center_y})，请切换到游戏画面...")
time.sleep(3)
adb_tap(center_x, center_y)
print("点击完成，观察游戏是否有反应。")
