# -*- coding: utf-8 -*-
import subprocess
import json
from pathlib import Path
from touch import ADBController

print("=== 诊断开始 ===")

# 1. 检查设备连接
print("\n[1] 检查 ADB 设备:")
result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
print(result.stdout.strip())

# 2. 读取校准文件
config_path = Path("configs/calibration.json")
if not config_path.exists():
    print("错误：校准文件不存在，请先运行 calibrate.py")
    exit()
with open(config_path, 'r') as f:
    calib = json.load(f)
window_w, window_h = calib['window_size']
move_stick = calib['calibration']['move_stick_center']
print(f"\n[2] 窗口尺寸: {window_w}x{window_h}")
print(f"移动摇杆归一化坐标: {move_stick}")

# 3. 计算手机绝对坐标
phone_w, phone_h = 1440, 3200
start_x_norm, start_y_norm = move_stick
start_win_x = int(start_x_norm * window_w)
start_win_y = int(start_y_norm * window_h)
start_phone_x = int(start_win_x * phone_w / window_w)
start_phone_y = int(start_win_y * phone_h / window_h)

# 向上滑动一段距离
end_norm_x = start_x_norm
end_norm_y = max(0.0, start_y_norm - 0.2)
end_win_x = int(end_norm_x * window_w)
end_win_y = int(end_norm_y * window_h)
end_phone_x = int(end_win_x * phone_w / window_w)
end_phone_y = int(end_win_y * phone_h / window_h)

print(f"\n[3] 计算的手机坐标:")
print(f"起点 (手机): {start_phone_x}, {start_phone_y}")
print(f"终点 (手机): {end_phone_x}, {end_phone_y}")
print(f"起点 (窗口): {start_win_x}, {start_win_y}")
print(f"终点 (窗口): {end_win_x}, {end_win_y}")

# 4. 尝试通过 ADBController 发送 swipe
print("\n[4] 尝试发送滑动指令...")
controller = ADBController(device_id="d1cbbc0f", phone_resolution=(phone_w, phone_h))
try:
    controller.swipe(
        start_x_norm, start_y_norm,
        end_norm_x, end_norm_y,
        duration=200,
        from_norm=True,
        win_w=window_w,
        win_h=window_h
    )
    print("swipe 调用完成，观察手机是否有滑动。")
except Exception as e:
    print(f"swipe 调用失败: {e}")

# 5. 直接执行 adb 命令作为对比
print("\n[5] 直接执行 adb 命令:")
cmd = f"adb shell input swipe {start_phone_x} {start_phone_y} {end_phone_x} {end_phone_y} 200"
print(f"执行: {cmd}")
subprocess.run(cmd, shell=True)
print("直接 adb 命令已执行。")

print("\n=== 诊断完成 ===")
