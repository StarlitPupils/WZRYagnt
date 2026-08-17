# -*- coding: utf-8 -*-
import time
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mss
import win32gui
import win32con
from touch import ADBController

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
            frame = frame[..., :3]
            return frame
        except:
            return None

class ZhongkuiFinal:
    def __init__(self, model_path, config_path, device_id="127.0.0.1:7555", conf_thres=0.3):
        self.model = YOLO(model_path)
        self.capture = ScreenCapture()
        self.controller = ADBController(device_id)
        self.conf_thres = conf_thres
        self.class_names = self.model.names
        with open(config_path, 'r') as f:
            calib = json.load(f)
        self.screen_w = calib['screen_width']
        self.screen_h = calib['screen_height']
        pts = calib['points']
        self.move_center = pts['move_stick_center']
        self.dir_vectors = {
            'up': pts['dir_up'],
            'left_up': pts['dir_left_up'],
            'right_up': pts['dir_right_up'],
            'left': pts['dir_left'],
            'right': pts['dir_right'],
            'left_down': pts['dir_left_down'],
            'right_down': pts['dir_right_down'],
            'down': pts['dir_down']
        }
        self.skill1 = pts['skill1']
        self.skill2 = pts['skill2']
        self.last_skill1_time = 0
        self.last_skill2_time = 0

    def get_state(self):
        frame = self.capture.get_frame()
        if frame is None:
            return None, None
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        detections = results[0]
        return frame, detections

    def act(self, detections, frame_shape):
        h, w = frame_shape[:2]
        hook_aim = None
        enemy_heroes = []
        minions = []

        if detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if self.class_names[cls_id] == 'hook_aim':
                    hook_aim = (center_x, center_y)
                elif self.class_names[cls_id] == 'enemy_hero':
                    enemy_heroes.append((center_x, center_y))
                elif self.class_names[cls_id] == 'minion':
                    minions.append((center_x, center_y))

        print(f"[DETECT] Hook: {hook_aim is not None} | Enemies: {len(enemy_heroes)} | Minions: {len(minions)}")

        if hook_aim is not None and len(enemy_heroes) > 0:
            now = time.time()
            if now - self.last_skill2_time > 2.0:
                x, y = self.skill2
                self.controller.tap(x, y)
                self.last_skill2_time = now
                print(">>> Skill 2 used (hook)")
                return
            else:
                print("Skill 2 on cooldown...")

        if enemy_heroes:
            center_x, center_y = w/2, h/2
            closest = min(enemy_heroes, key=lambda e: (e[0]-center_x)**2 + (e[1]-center_y)**2)
            dist = math.sqrt((closest[0]-center_x)**2 + (closest[1]-center_y)**2)
            if dist < 0.3 * w:
                now = time.time()
                if now - self.last_skill1_time > 1.5:
                    x, y = self.skill1
                    self.controller.tap(x, y)
                    self.last_skill1_time = now
                    print(f">>> Skill 1 used (dist={dist:.1f})")
                    return
                else:
                    print("Skill 1 on cooldown...")
            else:
                self._move_towards(closest[0], closest[1])
                print(f">>> Moving towards enemy ({closest[0]:.0f}, {closest[1]:.0f})")
                return

        if minions:
            closest = min(minions, key=lambda m: (m[0]-w/2)**2 + (m[1]-h/2)**2)
            self._move_towards(closest[0], closest[1])
            print(">>> Moving towards minion")
            return

        self._stop_move()
        print(">>> Idle")

    def _move_towards(self, target_x, target_y):
        start_x, start_y = self.move_center
        dx = target_x - start_x
        dy = target_y - start_y
        angle = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle)
        if -22.5 <= angle_deg < 22.5:
            dir_key = 'right'
        elif 22.5 <= angle_deg < 67.5:
            dir_key = 'right_up'
        elif 67.5 <= angle_deg < 112.5:
            dir_key = 'up'
        elif 112.5 <= angle_deg < 157.5:
            dir_key = 'left_up'
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            dir_key = 'left'
        elif -157.5 <= angle_deg < -112.5:
            dir_key = 'left_down'
        elif -112.5 <= angle_deg < -67.5:
            dir_key = 'down'
        else:
            dir_key = 'right_down'
        end_x, end_y = self.dir_vectors[dir_key]
        self.controller.swipe(start_x, start_y, end_x, end_y, duration=2000)
        print(f"    -> Holding direction: {dir_key} ({start_x},{start_y})->({end_x},{end_y})")

    def _stop_move(self):
        x, y = self.move_center
        self.controller.swipe(x, y, x, y, duration=50)
        print("    -> Stop moving")

    def run(self):
        print("Waiting for MuMu window...")
        while not self.capture.find_window():
            time.sleep(1)
        print("Zhongkui agent started. Press Ctrl+C to exit.")
        try:
            while True:
                frame, detections = self.get_state()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.act(detections, frame.shape)
                time.sleep(0.15)
        except KeyboardInterrupt:
            print("\nExited.")

if __name__ == "__main__":
    MODEL_PATH = "runs/detect/zhongkui_detector_finetune/weights/best.pt"
    CONFIG_PATH = "configs/calibration_absolute.json"
    DEVICE_ID = "127.0.0.1:7555"
    agent = ZhongkuiFinal(MODEL_PATH, CONFIG_PATH, DEVICE_ID)
    agent.run()
