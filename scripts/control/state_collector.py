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
from datetime import datetime

# ---------- 阵营识别 ----------
def detect_camp(frame, fountain_roi):
    """通过泉水区域主色调判断阵营"""
    x1, y1, x2, y2 = fountain_roi
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    avg_color = roi.mean(axis=(0, 1))
    b, g, r = avg_color
    if b > r * 1.2:
        return 'blue'
    elif r > b * 1.2:
        return 'red'
    else:
        return None

# ---------- 窗口截图 ----------
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

# ---------- 状态后处理 ----------
class StateProcessor:
    def __init__(self, camp=None):
        self.camp = camp

    def set_camp(self, camp):
        self.camp = camp

    def classify(self, generic_name, confidence):
        return generic_name, confidence

# ---------- 主采集器 ----------
class StateCollector:
    def __init__(self, model_path, save_dir="data/states"):
        self.model = YOLO(model_path)
        self.capture = ScreenCapture()
        self.processor = StateProcessor()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # 小地图中心硬编码为 (115, 115)
        mini_x, mini_y = 115, 115
        self.fountain_roi = (int(mini_x-15), int(mini_y+20), int(mini_x+15), int(mini_y+50))
        self.camp_detected = False

    def run(self, interval=0.5):
        print("Waiting for MuMu window...")
        while not self.capture.find_window():
            time.sleep(1)
        print("Starting state collection. Press Ctrl+C to stop.")
        frame_count = 0
        try:
            while True:
                frame = self.capture.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                if not self.camp_detected:
                    camp = detect_camp(frame, self.fountain_roi)
                    if camp:
                        self.processor.set_camp(camp)
                        self.camp_detected = True
                        print(f"Camp detected: {camp}")

                results = self.model(frame, conf=0.3, verbose=False)
                detections = results[0]

                state = {
                    'timestamp': datetime.now().isoformat(),
                    'camp': self.processor.camp,
                    'frame_id': frame_count,
                    'objects': []
                }

                if detections.boxes is not None:
                    boxes = detections.boxes.xyxy.cpu().numpy()
                    confs = detections.boxes.conf.cpu().numpy()
                    cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
                    names = self.model.names
                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        generic_name = names[cls_id]
                        final_name, _ = self.processor.classify(generic_name, conf)
                        x1, y1, x2, y2 = box.tolist()
                        state['objects'].append({
                            'class': final_name,
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1+x2)/2, (y1+y2)/2]
                        })

                filename = self.save_dir / f"state_{frame_count:06d}.json"
                with open(filename, 'w') as f:
                    json.dump(state, f)

                frame_count += 1
                print(f"\rSaved {frame_count} states", end='')
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nCollection finished. Total {frame_count} states saved.")

if __name__ == "__main__":
    MODEL_PATH = "runs/detect/zhongkui_detector_finetune/weights/best.pt"
    collector = StateCollector(MODEL_PATH)
    collector.run(interval=0.5)
