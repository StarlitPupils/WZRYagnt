# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

class MultiScaleTemplateExtractor:
    def __init__(self, template_path, angles=[0, 45, 90, 135, 180, 225, 270, 315], 
                 angle_names=['right', 'right_down', 'down', 'left_down', 
                              'left', 'left_up', 'up', 'right_up'], debug=False):
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError("Template not found")
        self.templates = {}
        self.angle_names = {}
        for ang, name in zip(angles, angle_names):
            # 旋转模板
            M = cv2.getRotationMatrix2D((self.template.shape[1]//2, self.template.shape[0]//2), ang, 1.0)
            rotated = cv2.warpAffine(self.template, M, (self.template.shape[1], self.template.shape[0]))
            self.templates[name] = rotated
        self.debug = debug

    def get_direction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_val = 0.5  # 阈值
        for direction, tmpl in self.templates.items():
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_match = (direction, max_loc, tmpl.shape)
        if best_match:
            direction, loc, shape = best_match
            if self.debug:
                h, w = shape
                cv2.rectangle(frame, loc, (loc[0]+w, loc[1]+h), (0,255,0), 2)
                cv2.putText(frame, direction, (loc[0], loc[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow('Template Matching', frame)
                cv2.waitKey(1)
            return direction
        return None

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        actions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            action = {'frame': frame_idx, 'type': 'none'}
            direction = self.get_direction(frame)
            if direction:
                action['type'] = 'move'
                action['direction'] = direction
            actions.append(action)
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()
        out_file = Path(output_dir) / f"{Path(video_path).stem}_actions.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(actions, f)
        print(f"Saved {len(actions)} actions to {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_template_actions.py <video> <output_dir> [--debug]")
        sys.exit(1)
    debug = '--debug' in sys.argv
    extractor = MultiScaleTemplateExtractor("configs/arrow_template.png", debug=debug)
    extractor.process_video(sys.argv[1], sys.argv[2])
