# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class ResultActionExtractor:
    def __init__(self, model_path, config_path):
        self.model = YOLO(model_path)
        with open(config_path, 'r') as f:
            calib = json.load(f)
        pts = calib['points']
        self.joystick_center = pts['move_stick_center']
        self.dir_vectors = {k: pts[k] for k in ['dir_up','dir_left_up','dir_right_up','dir_left','dir_right','dir_left_down','dir_right_down','dir_down']}

    def get_hero_position(self, results, frame_shape):
        if results.boxes is None:
            return None
        h, w = frame_shape[:2]
        center_x, center_y = w/2, h/2
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        hero_boxes = []
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            name = self.model.names[cls_id]
            if 'hero' in name.lower():
                hero_boxes.append((box, conf))
        if not hero_boxes:
            return None
        hero_boxes.sort(key=lambda x: ((x[0][0]+x[0][2])/2-center_x)**2 + ((x[0][1]+x[0][3])/2-center_y)**2)
        box = hero_boxes[0][0]
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    def _displacement_to_direction(self, dx, dy):
        # 使用余弦相似度匹配预设方向
        cx, cy = self.joystick_center
        best_dir = None
        best_sim = -2
        for dname, (tx, ty) in self.dir_vectors.items():
            vx = tx - cx
            vy = ty - cy
            norm = np.sqrt(vx*vx + vy*vy)
            if norm == 0:
                continue
            vx /= norm
            vy /= norm
            nx = dx / np.sqrt(dx*dx + dy*dy + 1e-5)
            ny = dy / np.sqrt(dx*dx + dy*dy + 1e-5)
            sim = nx * vx + ny * vy
            if sim > best_sim:
                best_sim = sim
                best_dir = dname
        return best_dir

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Can't open video: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        prev_pos = None
        actions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=0.3, verbose=False)[0]
            curr_pos = self.get_hero_position(results, frame.shape)
            action = {'frame': frame_idx, 'type': 'none'}
            if curr_pos and prev_pos:
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 5:
                    action['type'] = 'move'
                    action['direction'] = self._displacement_to_direction(dx, dy)
            actions.append(action)
            prev_pos = curr_pos
            frame_idx += 1
        cap.release()
        out_file = Path(output_dir) / f"{Path(video_path).stem}_actions.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(actions, f)
        print(f"Saved {len(actions)} actions to {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_result_actions.py <video_file> <output_dir>")
        sys.exit(1)
    extractor = ResultActionExtractor("runs/detect/zhongkui_detector_finetune/weights/best.pt", "configs/calibration_absolute.json")
    extractor.process_video(sys.argv[1], sys.argv[2])
