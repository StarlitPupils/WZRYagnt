import json
import sys
from pathlib import Path

def align_state_action(state_dir, action_file, output_file):
    # Load states by frame index
    states = {}
    state_path = Path(state_dir)
    for f in state_path.glob("*.json"):
        with open(f, 'r', encoding='utf-8') as fp:
            state = json.load(fp)
            frame_idx = state.get('frame_idx')
            if frame_idx is not None:
                states[frame_idx] = state

    # Load actions list
    with open(action_file, 'r', encoding='utf-8') as fp:
        actions = json.load(fp)

    # Align and write
    with open(output_file, 'w', encoding='utf-8') as out:
        for act in actions:
            frame = act.get('frame')
            if frame in states:
                sample = {
                    'state': states[frame],
                    'action': {k: v for k, v in act.items() if k != 'frame'}
                }
                out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Aligned {len(actions)} actions, output to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python align_state_action.py <state_dir> <action_file> <output_file>")
        sys.exit(1)
    align_state_action(sys.argv[1], sys.argv[2], sys.argv[3])
