# -*- coding: utf-8 -*-
import subprocess
import cv2
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np

class ScrcpyCapture:
    def __init__(self, scrcpy_path, device_id=None, max_fps=30, max_size=1024, bit_rate='8M'):
        self.scrcpy_path = Path(scrcpy_path)
        self.scrcpy_exe = self.scrcpy_path / "scrcpy.exe"
        if not self.scrcpy_exe.exists():
            raise FileNotFoundError(f"scrcpy.exe not found at {self.scrcpy_exe}")
        
        self.device_id = device_id
        self.max_fps = max_fps
        self.max_size = max_size
        self.bit_rate = bit_rate
        self.process = None
        self.temp_video = None
        self.cap = None
        
    def start(self):
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=r"E:\WZRYagent\temp")
        self.video_path = Path(self.temp_video.name)
        
        cmd = [
            str(self.scrcpy_exe),
            '--no-audio',
            '--no-window',
            '--no-control',
            '--max-fps', str(self.max_fps),
            '--max-size', str(self.max_size),
            '--video-bit-rate', self.bit_rate,
            '--record', str(self.video_path)
        ]
        
        print(f"Start cmd: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, cwd=self.scrcpy_path)
        
        for _ in range(10):
            if self.video_path.exists():
                break
            time.sleep(0.5)
        
        if not self.video_path.exists():
            raise RuntimeError("Video file not generated, check scrcpy.")
        
        time.sleep(1)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video stream.")
            
        print("Video stream started.")
        return self

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def stop(self):
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.temp_video:
            self.temp_video.close()
            try:
                os.unlink(self.temp_video.name)
            except:
                pass
        print("Resources released.")

class ScreenshotCollector:
    def __init__(self, hero_name="zhongkui", save_dir="data/screenshots"):
        self.hero_name = hero_name
        self.save_dir = Path(save_dir) / hero_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.capture = None

    def collect(self, interval=1.0, max_count=None):
        count = 0
        print(f"Collecting every {interval} sec, saving to {self.save_dir}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                frame = self.capture.get_frame()
                if frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.hero_name}_{timestamp}_{count:06d}.jpg"
                    filepath = self.save_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    count += 1
                    print(f"[{count}] Saved: {filename}")
                else:
                    time.sleep(0.01)
                if max_count and count >= max_count:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nUser interrupted.")
        finally:
            print(f"Finished, {count} images saved.")

if __name__ == "__main__":
    SCRCPY_PATH = "E:/WZRYagent/tools/scrcpy"
    HERO_NAME = "zhongkui"
    INTERVAL = 1.0

    cap = ScrcpyCapture(SCRCPY_PATH)
    cap.start()

    collector = ScreenshotCollector(hero_name=HERO_NAME)
    collector.capture = cap
    try:
        collector.collect(interval=INTERVAL)
    finally:
        cap.stop()
