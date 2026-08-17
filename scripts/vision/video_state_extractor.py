# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime

class VideoStateExtractor:
    def __init__(self, model_path, ocr_enabled=False):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(lang='en') if ocr_enabled else None
        # UI区域（根据你的视频分辨率调整）
        self.ui_regions = {
            'hp': (50, 680, 150, 710),
            'skill1_cd': (900, 620, 940, 640),
            'skill2_cd': (980, 620, 1020, 640)
        }

    def extract_ui_text(self, frame):
        ui_data = {}
        if self.ocr is None:
            return ui_data
        for name, (x1, y1, x2, y2) in self.ui_regions.items():
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            result = self.ocr.predict(thresh)
            if result and isinstance(result, list):
                texts = []
                for item in result:
                    if hasattr(item, 'rec_texts'):
                        texts.extend(item.rec_texts)
                    elif isinstance(item, dict) and 'rec_texts' in item:
                        texts.extend(item['rec_texts'])
                ui_data[name] = ''.join(texts).strip()
        return ui_data

    def process_video(self, video_path, output_dir, sample_interval=0.5):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Can't open video: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * sample_interval))
        frame_idx = 0
        saved_count = 0
        output_path = Path(output_dir) / Path(video_path).stem
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {video_path} (FPS: {fps}, interval: {frame_interval} frames)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            results = self.model(frame, conf=0.3, verbose=False)
            detections = results[0]
            ui_texts = self.extract_ui_text(frame)

            state = {
                'video': Path(video_path).name,
                'frame_idx': frame_idx,
                'timestamp': datetime.now().isoformat(),
                'ui': ui_texts,
                'objects': []
            }

            if detections.boxes is not None:
                boxes = detections.boxes.xyxy.cpu().numpy()
                confs = detections.boxes.conf.cpu().numpy()
                cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
                names = self.model.names
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    x1, y1, x2, y2 = box.tolist()
                    state['objects'].append({
                        'class': names[cls_id],
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1+x2)/2, (y1+y2)/2]
                    })

            state_file = output_path / f"frame_{frame_idx:06d}.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            saved_count += 1
            print(f"\rSaved {saved_count} states", end='')
            frame_idx += 1

        cap.release()
        print(f"\nDone! {saved_count} states saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python video_state_extractor.py <video_file_or_dir> <output_dir>")
        sys.exit(1)
    video_input = sys.argv[1]
    output_dir = sys.argv[2]
    extractor = VideoStateExtractor("runs/detect/zhongkui_detector_finetune/weights/best.pt", ocr_enabled=False)
    video_path = Path(video_input)
    if video_path.is_file():
        extractor.process_video(video_path, output_dir)
    elif video_path.is_dir():
        for vid in video_path.glob("*.mp4"):
            extractor.process_video(vid, output_dir)
    else:
        print("Invalid input path")
