# -*- coding: utf-8 -*-
import cv2
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mss
import win32gui
import win32con
from mumu import Mumu

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except:
            return None

class ZhongkuiAgent:
    def __init__(self, model_path, mumu_manager_path, config_path, conf_thres=0.3):
        self.model = YOLO(model_path)
        self.capture = ScreenCapture()
        self.mumu = Mumu(mumu_manager_path).select(1)  # 选择第一个模拟器实例
        self.conf_thres = conf_thres
        self.class_names = self.model.names
        with open(config_path, 'r') as f:
            self.calib = json.load(f)
        self.move_stick_center = self.calib['move_stick_center']
        self.skill1_pos = self.calib['skill1']
        self.skill2_pos = self.calib['skill2']
        self.last_skill1_time = 0
        self.window_created = False

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

        # 打印检测结果
        objs = []
        if detections.boxes is not None:
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                objs.append(f"{self.class_names[cls_id]} ({conf:.2f})")
        if objs:
            print(f"检测到: {', '.join(objs)}")
        else:
            print("无目标")

        if hook_aim and enemy_heroes:
            self._tap_skill2(w, h)
            print("释放二技能！")
            return

        if enemy_heroes:
            closest = min(enemy_heroes, key=lambda e: (e[0]-w/2)**2 + (e[1]-h/2)**2)
            dist = np.sqrt((closest[0]-w/2)**2 + (closest[1]-h/2)**2)
            if dist < 0.3 * w:
                now = time.time()
                if now - self.last_skill1_time > 1.5:
                    self._tap_skill1(w, h)
                    self.last_skill1_time = now
                    print("释放一技能！")
                else:
                    print("一技能冷却中...")
                return
            else:
                self._move_towards(closest[0], closest[1], w, h)
                print("向敌人移动")
                return

        if minions:
            closest = min(minions, key=lambda m: (m[0]-w/2)**2 + (m[1]-h/2)**2)
            self._move_towards(closest[0], closest[1], w, h)
            print("向小兵移动")
            return

        print("待机")

    def _denorm(self, norm_x, norm_y, w, h):
        return int(norm_x * w), int(norm_y * h)

    def _tap_skill2(self, w, h):
        x, y = self._denorm(self.skill2_pos[0], self.skill2_pos[1], w, h)
        self.mumu.adb.tap(x, y)

    def _tap_skill1(self, w, h):
        x, y = self._denorm(self.skill1_pos[0], self.skill1_pos[1], w, h)
        self.mumu.adb.tap(x, y)

    def _move_towards(self, target_x, target_y, w, h):
        start_x, start_y = self._denorm(self.move_stick_center[0], self.move_stick_center[1], w, h)
        dx = target_x - start_x
        dy = target_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-5:
            return
        step = 120
        dx_norm = dx / length
        dy_norm = dy / length
        end_x = int(start_x + dx_norm * step)
        end_y = int(start_y + dy_norm * step)
        end_x = max(0, min(w-1, end_x))
        end_y = max(0, min(h-1, end_y))
        self.mumu.adb.swipe(start_x, start_y, end_x, end_y, 100)

    def run(self, display=True):
        print("等待 MuMu 模拟器窗口...")
        while not self.capture.find_window():
            time.sleep(1)
        print("钟馗智能体启动。按 'q' 退出。")
        cv2.namedWindow('Zhongkui Agent', cv2.WINDOW_NORMAL)
        self.window_created = True
        # 获取一帧确定尺寸
        frame, _ = self.get_state()
        if frame is not None:
            h, w = frame.shape[:2]
            cv2.resizeWindow('Zhongkui Agent', w, h)

        while True:
            frame, detections = self.get_state()
            if frame is None:
                time.sleep(0.05)
                continue

            self.act(detections, frame.shape)

            if display:
                annotated = detections.plot() if detections.boxes is not None else frame
                cv2.imshow('Zhongkui Agent', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.15)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "runs/detect/zhongkui_detector_finetune/weights/best.pt"
    MUMU_MANAGER_PATH = r"E:\MuMuPlayer\nx_main\MuMuManager.exe"
    CONFIG_PATH = "configs/calibration.json"
    agent = ZhongkuiAgent(MODEL_PATH, MUMU_MANAGER_PATH, CONFIG_PATH)
    agent.run(display=True)
