# -*- coding: utf-8 -*-
import cv2
import time
import json
import numpy as np
import subprocess
from pathlib import Path
from ultralytics import YOLO
import mss
import pygetwindow as gw

class ScreenCapture:
    def __init__(self, window_title="scrcpy", rotate_clockwise=False):
        self.window_title = window_title
        self.rotate_clockwise = rotate_clockwise
        self.sct = mss.mss()
        self.window_rect = None

    def find_window(self):
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            return None
        win = windows[0]
        if win.isMinimized:
            win.restore()
        left = win.left + 8
        top = win.top + 30
        width = win.width - 16
        height = win.height - 38
        return {"left": left, "top": top, "width": width, "height": height}

    def get_frame(self):
        if self.window_rect is None:
            self.window_rect = self.find_window()
            if self.window_rect is None:
                return None
        try:
            screenshot = self.sct.grab(self.window_rect)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if self.rotate_clockwise:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            return frame
        except Exception as e:
            return None

class RealTimeDetector:
    def __init__(self, model_path, window_title="scrcpy", conf_thres=0.5, rotate_clockwise=False):
        self.model = YOLO(model_path)
        self.capture = ScreenCapture(window_title, rotate_clockwise)
        self.conf_thres = conf_thres
        self.class_names = self.model.names

    def run(self, display=True, save_state=False, max_display_height=900):
        print("等待 scrcpy 窗口...")
        while self.capture.find_window() is None:
            time.sleep(1)
        print("开始实时检测。按 'q' 退出。")
        cv2.namedWindow('Zhongkui Detector', cv2.WINDOW_NORMAL)
        frame_count = 0
        while True:
            start_time = time.time()
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            results = self.model(frame, conf=self.conf_thres, verbose=False)
            detections = results[0]

            if display:
                annotated = detections.plot()
                h, w = annotated.shape[:2]
                scale = max_display_height / h
                new_w = int(w * scale)
                new_h = int(h * scale)
                annotated = cv2.resize(annotated, (new_w, new_h))
                cv2.imshow('Zhongkui Detector', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            fps = 1.0 / (time.time() - start_time)
            print(f"\r帧 {frame_count} | FPS: {fps:.2f}", end='')

        cv2.destroyAllWindows()

    def _extract_state(self, detections, img_shape):
        h, w = img_shape[:2]
        state = {'timestamp': time.time(), 'image_size': [w, h], 'objects': []}
        if detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box.tolist()
                state['objects'].append({
                    'class': self.class_names[cls_id],
                    'class_id': cls_id,
                    'confidence': float(conf),
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'center': [(x1+x2)/(2*w), (y1+y2)/(2*h)]
                })
        return state

if __name__ == "__main__":
    MODEL_PATH = "runs/detect/zhongkui_detector_finetune/weights/best.pt"
    # 不旋转，直接显示 scrcpy 窗口原始内容
    detector = RealTimeDetector(
        model_path=MODEL_PATH,
        window_title="scrcpy",
        conf_thres=0.5,
        rotate_clockwise=False
    )
    detector.run(display=True, save_state=False, max_display_height=900)
