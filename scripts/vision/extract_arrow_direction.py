# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path

class ArrowDirectionExtractor:
    def __init__(self, debug=False, delay=100):
        self.debug = debug
        self.delay = delay  # 毫秒
        self.lower = np.array([0, 0, 200])
        self.upper = np.array([180, 30, 255])
        self.area_min = 50
        self.area_max = 2000
        self.aspect_min = 2.0
        self.dir_vectors = {
            'right': (1, 0),
            'right_up': (0.707, -0.707),
            'up': (0, -1),
            'left_up': (-0.707, -0.707),
            'left': (-1, 0),
            'left_down': (-0.707, 0.707),
            'down': (0, 1),
            'right_down': (0.707, 0.707)
        }

    def get_arrow_direction(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        arrow_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.area_min or area > self.area_max:
                continue
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w < 5 or h < 5:
                continue
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect < self.aspect_min:
                continue
            arrow_candidates.append((c, rect, area))
        
        if not arrow_candidates:
            return None
        
        best_c = max(arrow_candidates, key=lambda x: x[2])[0]
        rect = cv2.minAreaRect(best_c)
        angle = rect[2]
        w, h = rect[1]
        if w > h:
            dx = np.cos(np.radians(angle))
            dy = np.sin(np.radians(angle))
        else:
            dx = -np.sin(np.radians(angle))
            dy = np.cos(np.radians(angle))
        
        norm = np.sqrt(dx*dx + dy*dy)
        if norm == 0:
            return None
        dx, dy = dx/norm, dy/norm
        dx, dy = -dx, -dy  # 修正方向反转
        
        best_dir = None
        best_sim = -2
        for dname, (vx, vy) in self.dir_vectors.items():
            sim = dx*vx + dy*vy
            if sim > best_sim:
                best_sim = sim
                best_dir = dname
        
        if self.debug:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(frame, [box], 0, (0,255,0), 2)
            cv2.putText(frame, best_dir, (int(rect[0][0]), int(rect[0][1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Arrow Detection', frame)
            if cv2.waitKey(self.delay) & 0xFF == ord('q'):
                return None
        return best_dir

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
            direction = self.get_arrow_direction(frame)
            if direction:
                action['type'] = 'move'
                action['direction'] = direction
            actions.append(action)
            frame_idx += 1
            if not self.debug:
                print(f"\rProcessing frame {frame_idx}", end='')
        cap.release()
        cv2.destroyAllWindows()
        out_file = Path(output_dir) / f"{Path(video_path).stem}_actions.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(actions, f)
        print(f"\nSaved {len(actions)} actions to {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_arrow_direction.py <video> <output_dir> [--debug] [--delay MS]")
        sys.exit(1)
    debug = '--debug' in sys.argv
    delay = 100
    for i, arg in enumerate(sys.argv):
        if arg == '--delay' and i+1 < len(sys.argv):
            delay = int(sys.argv[i+1])
    extractor = ArrowDirectionExtractor(debug=debug, delay=delay)
    extractor.process_video(sys.argv[1], sys.argv[2])
