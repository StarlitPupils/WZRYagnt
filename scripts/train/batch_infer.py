# -*- coding: utf-8 -*-
"""批量录像动作反推（M3 数据量产入口）。

对 data/raw_videos/ 下所有带 HUD 的对局录像执行 infer_actions，
输出每部录像的 actions.json（含事件与统计），并生成汇总索引。

用法：
    venv\\Scripts\\python.exe scripts\\train\\batch_infer.py
        [--dir data/raw_videos] [--calib configs/calibration_absolute.json]
        [--out-dir data/actions] [--sample-every 3] [--yolo-every 12]
        [--model runs/detect/zhongkui_11cls/weights/best.pt]
        [--fix-mkv]   # 用 ffmpeg 修复损坏 mkv 后处理（scrcpy 录制被杀时 mkv 尾部不完整）

说明：录像建议先用 ffmpeg 修复（scrcpy 录制可能损坏）；修复流程见 docs/PLAN.md。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VIDEO_EXTS = (".mp4", ".mkv", ".avi")


def fix_video(path: Path, out: Path) -> Path:
    """ffmpeg 重封装修复损坏 mkv/mp4，返回修复后路径。"""
    r = subprocess.run(["ffmpeg", "-y", "-i", str(path), "-c", "copy",
                        "-movflags", "+faststart", str(out)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg 修复失败: {path}\n{r.stderr[-500:]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "raw_videos"))
    ap.add_argument("--calib", default=str(ROOT / "configs" / "calibration_absolute.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "actions"))
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--yolo-every", type=int, default=12)
    ap.add_argument("--model", default=str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"))
    ap.add_argument("--fix-mkv", action="store_true", help="处理前用 ffmpeg 修复损坏 mkv")
    args = ap.parse_args()

    from wzry.train.action_infer import infer_actions, split_result

    video_dir = Path(args.dir)
    if not video_dir.exists():
        print(f"录像目录不存在: {video_dir}")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in video_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"{video_dir} 下没有录像")
        return 1

    index = []
    for v in videos:
        print(f"\n=== {v.name} ===")
        src = v
        if args.fix_mkv and v.suffix.lower() == ".mkv":
            fixed = out_dir / f"_fixed_{v.stem}.mp4"
            if not fixed.exists():
                fix_video(v, fixed)
            src = fixed
        result = infer_actions(str(src), args.calib, sample_every=args.sample_every,
                               model_path=args.model, yolo_every=args.yolo_every,
                               progress=False)
        actions, meta = split_result(result)
        stats = meta.get("stats", {})
        out_file = out_dir / f"{v.stem}_actions.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"video": v.name, "actions": actions, "events": meta.get("events", []),
                       "stats": stats}, f, ensure_ascii=False, indent=1)
        n_move = sum(1 for a in actions if a["type"] == "move")
        n_skill = sum(1 for a in actions if a["type"] == "skill")
        index.append({"video": v.name, "actions_file": str(out_file),
                      "move": n_move, "skill": n_skill, "frames": stats.get("frames_processed")})
        print(f"  move={n_move} skill={n_skill} -> {out_file.name}")

    idx_file = out_dir / "index.json"
    idx_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总索引: {idx_file}（{len(videos)} 部录像）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
