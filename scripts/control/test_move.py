# -*- coding: utf-8 -*-
import json
from touch import ADBController

# 加载校准配置
with open("configs/calibration.json", 'r') as f:
    calib = json.load(f)

window_w, window_h = calib['window_size']
move_stick = calib['calibration']['move_stick_center']

controller = ADBController(device_id="d1cbbc0f", phone_resolution=(1440, 3200))

print("移动方向测试：")
print("输入方向指令（w/a/s/d），按 q 退出。")

while True:
    cmd = input("方向 (w/a/s/d) 或 q: ").strip().lower()
    if cmd == 'q':
        break
    if cmd not in ['w', 'a', 's', 'd']:
        print("无效输入")
        continue

    # 定义归一化方向向量
    direction_map = {
        'w': (0, -0.2),   # 上
        's': (0, 0.2),    # 下
        'a': (-0.2, 0),   # 左
        'd': (0.2, 0)     # 右
    }
    dx, dy = direction_map[cmd]

    start_x, start_y = move_stick
    end_x = max(0.0, min(1.0, start_x + dx))
    end_y = max(0.0, min(1.0, start_y + dy))

    print(f"滑动: 从 ({start_x:.2f},{start_y:.2f}) 到 ({end_x:.2f},{end_y:.2f})")
    controller.swipe(start_x, start_y, end_x, end_y, duration=100,
                     from_norm=True, win_w=window_w, win_h=window_h)
