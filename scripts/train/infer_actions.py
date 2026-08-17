# -*- coding: utf-8 -*-
"""动作反推器 v2 CLI：从带 HUD 对局录像推断玩家动作（多通道 + 交叉验证 + 置信度）。

用法:
  python scripts/train/infer_actions.py --video xxx.mp4 --calib xxx.calibration.json \
      --out actions.json --sample-every 1
  python scripts/train/infer_actions.py --video xxx.mp4 --sample-every 3 --yolo-every 3 \
      --debug-viz temp/infer_debug --max-frames 50

说明:
  - 校准文件缺省时按顺序自动查找：<video>.calibration.json -> configs/calibration.json
    -> configs/calibration_absolute.json（后两者为归一化/绝对坐标，按视频实际分辨率换算）。
  - 输出 JSON：{"video","calib","model","sample_every","fps","actions":[...],
    "events":[...],"stats":{...}}。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wzry.train.action_infer import (  # noqa: E402
    DEFAULT_MODEL,
    infer_actions,
    split_result,
)


def resolve_calib(video_path, calib_arg):
    if calib_arg:
        return calib_arg
    cands = [
        Path(str(video_path) + ".calibration.json"),
        Path(video_path).with_suffix(".calibration.json"),
        Path("configs/calibration.json"),
        Path("configs/calibration_absolute.json"),
    ]
    for c in cands:
        if c.exists():
            print(f"[calib] 使用 {c}")
            return str(c)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="动作反推器 v2（M3 数据工厂核心）")
    ap.add_argument("--video", required=True, help="带 HUD 的对局录像")
    ap.add_argument("--calib", default=None, help="校准文件（calibrate_video.py 输出）")
    ap.add_argument("--out", default="actions.json", help="输出 JSON 路径")
    ap.add_argument("--sample-every", type=int, default=1, help="抽样帧间隔，默认 1=每帧")
    ap.add_argument("--model", default=None,
                    help=f"YOLO 权重（默认 {DEFAULT_MODEL}，不存在则关闭通道 C）")
    ap.add_argument("--yolo-every", type=int, default=1, help="YOLO 运行间隔（相对采样帧）")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理采样帧数")
    ap.add_argument("--debug-viz", default=None, help="调试帧输出目录")
    ap.add_argument("--debug-every", type=int, default=1, help="调试帧保存间隔")
    ap.add_argument("--emit-none", action="store_true", help="输出 type=none 的帧动作")
    ap.add_argument("--emit-aim-only", action="store_true",
                    help="输出仅有 D 瞄准线（无按钮按下）的拖瞄事件")
    ap.add_argument("--c-window", type=float, default=0.18, help="C/D 证据匹配时间窗(秒)")
    ap.add_argument("--fps", type=float, default=None, help="覆盖视频 fps")
    ap.add_argument("--quiet", action="store_true", help="关闭进度输出")
    args = ap.parse_args(argv)

    calib = resolve_calib(args.video, args.calib)
    if calib is None:
        print("错误：未找到校准文件，请先运行 scripts/vision/calibrate_video.py 或用 --calib 指定")
        return 2

    model_path = args.model
    if model_path and model_path.lower() == "none":
        model_path = None
        print("[model] 通道 C 关闭（--model none）")
    elif model_path is None and DEFAULT_MODEL.exists():
        model_path = str(DEFAULT_MODEL)
        print(f"[model] 使用默认模型 {model_path}")
    elif model_path is not None:
        if not Path(model_path).exists():
            print(f"警告：模型不存在 {model_path}，通道 C 关闭")
            model_path = None

    print(f"[infer] 视频={args.video} calib={calib} sample_every={args.sample_every}"
          f" model={model_path or 'None(通道C关闭)'}")
    result = infer_actions(
        args.video, calib, sample_every=args.sample_every, model_path=model_path,
        debug_viz=args.debug_viz, emit_none=args.emit_none,
        yolo_every=args.yolo_every, max_frames=args.max_frames,
        c_window_s=args.c_window, emit_aim_only=args.emit_aim_only,
        progress=not args.quiet, fps_override=args.fps,
    )
    actions, meta = split_result(result)
    events = meta.get("events", [])
    stats = meta.get("stats", {})

    out = {
        "video": str(args.video),
        "calib": calib,
        "model": model_path,
        "sample_every": args.sample_every,
        "fps": stats.get("fps"),
        "frames_processed": stats.get("frames_processed"),
        "actions": actions,
        "events": events,
        "stats": stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[out] 写入 {out_path}（动作 {len(actions)} 条，原始事件 {len(events)} 条）")
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}")
    n_move = sum(1 for a in actions if a["type"] == "move")
    n_skill = sum(1 for a in actions if a["type"] == "skill")
    n_attack = sum(1 for a in actions if a["type"] == "attack")
    print(f"[summary] move={n_move} skill={n_skill} attack={n_attack} 总={len(actions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
