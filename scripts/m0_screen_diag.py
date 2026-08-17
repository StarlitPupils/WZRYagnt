# -*- coding: utf-8 -*-
"""M0 画面诊断：OCR 识别当前界面 + 主色统计 + 双 ADB 设备输入验证。

用法：
    venv\\Scripts\\python.exe scripts\\m0_screen_diag.py [--ocr] [--tap-test]

  --ocr      用 PaddleOCR 识别画面文字（首次运行可能加载模型较慢）
  --tap-test 在两个 ADB 设备上各点击一次 minimap_center 附近的"高对比点"，
             比较点击前后画面变化，判定哪个设备真正接收输入
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.capture.window import WindowCapture  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--tap-test", action="store_true")
    args = ap.parse_args()

    cap = WindowCapture()
    frame, ms = cap.get_frame()
    if frame is None:
        print("未找到 MuMu 窗口")
        return
    h, w = frame.shape[:2]
    print(f"窗口画面: {w}x{h} (采集 {ms:.0f} ms)")

    # 保存全尺寸截图供人工查看
    out = ROOT / "temp" / "screen_diag.png"
    cv2.imwrite(str(out), frame)
    print(f"截图已保存: {out}")

    # 4 分块平均色
    print("\n[分块平均色 BGR]")
    for r_i, name_r in enumerate(("上", "下")):
        for c_i, name_c in enumerate(("左", "右")):
            blk = frame[r_i * h // 2:(r_i + 1) * h // 2, c_i * w // 2:(c_i + 1) * w // 2]
            b, g, r = blk.reshape(-1, 3).mean(axis=0)
            print(f"  {name_r}{name_c}: ({b:6.1f}, {g:6.1f}, {r:6.1f})")

    # minimap ROI 色点统计
    cx, cy = int(0.086 * w), int(0.129 * h)
    r_mini = int(0.082 * min(w, h))
    roi = frame[cy - r_mini:cy + r_mini, cx - r_mini:cx + r_mini]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "蓝": cv2.inRange(hsv, (100, 80, 40), (130, 255, 255)),
        "红": cv2.inRange(hsv, (0, 80, 40), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 40), (180, 255, 255)),
        "黄": cv2.inRange(hsv, (20, 80, 40), (35, 255, 255)),
    }
    print("\n[minimap ROI 色点统计]")
    for name, m in masks.items():
        print(f"  {name}: {int((m > 0).sum())} 像素")

    if args.ocr:
        print("\n[OCR 识别画面文字]")
        try:
            from paddleocr import PaddleOCR
            t0 = time.time()
            ocr = PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                            use_doc_unwarping=False, use_textline_orientation=False)
            print(f"  PaddleOCR 加载耗时 {time.time()-t0:.1f}s")
            t0 = time.time()
            res = ocr.predict(frame)
            print(f"  识别耗时 {time.time()-t0:.1f}s")
            texts = []
            for page in res:
                for line in page.get("rec_texts", []) or []:
                    texts.append(line)
            print(f"  识别到 {len(texts)} 条文字:")
            for t in texts[:40]:
                print(f"    - {t}")
            if not texts:
                print("    (无文字，画面可能为纯场景/加载图)")
        except Exception as e:
            print(f"  OCR 失败: {e}")

    if args.tap_test:
        print("\n[双设备输入验证]")
        import subprocess
        from wzry.control.executor import discover_mumu_adb_devices
        serials = discover_mumu_adb_devices()
        print(f"  设备: {serials}")
        for serial in serials:
            before, _ = cap.get_frame()
            # 点一个小地图附近点：任何界面点击若有反馈都会变
            tx, ty = int(0.15 * w), int(0.30 * h)
            import subprocess as sp
            r = sp.run(["adb", "-s", serial, "shell", "input", "tap", str(tx), str(ty)],
                       capture_output=True, text=True, timeout=3)
            time.sleep(0.3)
            after, _ = cap.get_frame()
            diff = float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
            print(f"  {serial:<18} tap({tx},{ty}) 全画面变化 {diff:6.2f} (rc={r.returncode})")

    cap.close()


if __name__ == "__main__":
    main()
