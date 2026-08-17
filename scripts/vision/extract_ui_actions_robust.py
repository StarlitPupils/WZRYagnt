# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

class RobustUIActionExtractor:
    def __init__(self, calib_path):
        with open(calib_path, 'r') as f:
            self.calib = json.load(f)
        self.pts = self.calib['points']
        self.vid_w, self.vid_h = self.calib['video_resolution']

    def get_arrow_direction(self, frame):
        """通过模板匹配或颜色检测找到方向箭头，返回其指向"""
        # 简化：检测白色箭头区域，计算其最小外接矩形的方向
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        # 假设箭头是一个细长的白色区域，找到轮廓后计算方向
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # 取面积最大的轮廓
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 20:
            return None
        # 计算轮廓的最小外接矩形，获取角度
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        # 将OpenCV角度转换为方向标签（需结合箭头实际指向校准）
        # 简化：假设角度0°为向右，90°为向上
        if -22.5 <= angle < 22.5: return 'right'
        elif 22.5 <= angle < 67.5: return 'right_up'
        elif 67.5 <= angle < 112.5: return 'up'
        elif 112.5 <= angle < 157.5: return 'left_up'
        elif angle >= 157.5 or angle < -157.5: return 'left'
        elif -157.5 <= angle < -112.5: return 'left_down'
        elif -112.5 <= angle < -67.5: return 'down'
        else: return 'right_down'

    def is_button_pressed(self, frame, btn_pos):
        """检测按钮是否高亮（按下）"""
        x, y = btn_pos
        roi = frame[y-15:y+15, x-15:x+15]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 按下时通常变亮或出现特效，这里用亮度阈值
        return np.mean(gray) > 200

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        frame_idx = 0
        actions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            action = {'frame': frame_idx, 'type': 'none'}
            # 检测移动
            direction = self.get_arrow_direction(frame)
            if direction:
                action['type'] = 'move'
                action['direction'] = direction
            else:
                # 检测技能
                if self.is_button_pressed(frame, self.pts['skill1']):
                    action['type'] = 'skill'
                    action['skill_id'] = 1
                elif self.is_button_pressed(frame, self.pts['skill2']):
                    action['type'] = 'skill'
                    action['skill_id'] = 2
                elif self.is_button_pressed(frame, self.pts['skill3']):
                    action['type'] = 'skill'
                    action['skill_id'] = 3
                elif self.is_button_pressed(frame, self.pts['attack']):
                    action['type'] = 'attack'
                elif self.is_button_pressed(frame, self.pts['recall']):
                    action['type'] = 'recall'
                elif self.is_button_pressed(frame, self.pts['summoner']):
                    action['type'] = 'summoner'
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
        print("Usage: python extract_ui_actions_robust.py <video> <output_dir>")
        sys.exit(1)
    video = sys.argv[1]
    calib = Path(video).with_suffix('.calibration.json')
    extractor = RobustUIActionExtractor(str(calib))
    extractor.process_video(video, sys.argv[2])
