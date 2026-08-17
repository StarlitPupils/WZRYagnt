# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

class UIActionExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            calib = json.load(f)
        pts = calib['points']
        self.joystick_center = pts['move_stick_center']
        # 方向映射：根据摇杆小球相对于中心的位置
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
        self.skill1_pos = pts['skill1']
        self.skill2_pos = pts['skill2']

    def _pos_to_direction(self, pos):
        cx, cy = self.joystick_center
        dx = pos[0] - cx
        dy = pos[1] - cy
        # 计算与各预设方向向量的夹角余弦，选择最接近的
        best_dir = None
        best_sim = -2
        for dname, (tx, ty) in self.dir_vectors.items():
            vx = tx - cx
            vy = ty - cy
            # 归一化
            norm = np.sqrt(vx*vx + vy*vy)
            if norm == 0:
                continue
            vx /= norm
            vy /= norm
            # 当前向量
            nx = dx / np.sqrt(dx*dx + dy*dy + 1e-5)
            ny = dy / np.sqrt(dx*dx + dy*dy + 1e-5)
            sim = nx * vx + ny * vy  # 余弦相似度
            if sim > best_sim:
                best_sim = sim
                best_dir = dname
        return best_dir

    def get_joystick_position(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None

    def get_skill_pressed(self, frame, skill_pos):
        x, y = skill_pos
        roi = frame[y-10:y+10, x-10:x+10]
        if roi.size == 0:
            return False
        brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        return brightness > 200

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Can't open video: {video_path}")
            return
        frame_idx = 0
        actions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            action = {'frame': frame_idx, 'type': 'none'}
            pos = self.get_joystick_position(frame)
            if pos:
                action['type'] = 'move'
                action['direction'] = self._pos_to_direction(pos)
            elif self.get_skill_pressed(frame, self.skill1_pos):
                action['type'] = 'skill'
                action['skill_id'] = 1
            elif self.get_skill_pressed(frame, self.skill2_pos):
                action['type'] = 'skill'
                action['skill_id'] = 2
            actions.append(action)
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
        print("Usage: python extract_ui_actions.py <video_file> <output_dir>")
        sys.exit(1)
    extractor = UIActionExtractor("configs/calibration_absolute.json")
    extractor.process_video(sys.argv[1], sys.argv[2])
