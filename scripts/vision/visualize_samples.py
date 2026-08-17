import json
import cv2
import random
from pathlib import Path

def visualize_sample(video_path, state_dir, action_file, output_dir, num_samples=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 加载动作列表
    with open(action_file, 'r') as f:
        actions = json.load(f)
    
    # 随机选择帧索引
    state_files = list(Path(state_dir).glob("*.json"))
    sampled = random.sample(state_files, min(num_samples, len(state_files)))
    
    for sf in sampled:
        with open(sf, 'r') as f:
            state = json.load(f)
        frame_idx = state['frame_idx']
        # 找到对应的动作
        action = actions[frame_idx] if frame_idx < len(actions) else None
        if action is None:
            continue
        
        # 读取视频帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 在画面上绘制动作信息
        text = f"Action: {action['type']}"
        if action['type'] == 'move':
            text += f" {action.get('direction', '')}"
        elif action['type'] == 'skill':
            text += f" {action.get('skill_id', '')}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # 同时绘制检测到的目标框（可选）
        for obj in state.get('objects', []):
            x1,y1,x2,y2 = obj['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)
        
        out_file = output_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_file), frame)
        print(f"Saved {out_file}")
    cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python visualize_samples.py <video> <state_dir> <action_file> <output_dir>")
        sys.exit(1)
    visualize_sample(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
