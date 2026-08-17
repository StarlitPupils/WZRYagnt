# -*- coding: utf-8 -*-
"""Interactive YOLO box annotator for zhongkui screenshots (11 classes).

Supports loading existing same-name .txt labels on navigation and via 'l'.

Controls:
    n        next image (auto-saves if new boxes were drawn)
    p        previous image (auto-saves if new boxes were drawn)
    c        cycle current class
    s        save current labels to <image>.txt
    l        (re)load existing labels from <image>.txt
    d        delete last drawn box
    q        quit
"""
import cv2
import sys
from pathlib import Path

CLASSES = [
    'enemy_hero', 'ally_hero', 'enemy_minion', 'ally_minion',
    'enemy_turret', 'ally_turret', 'enemy_crystal', 'ally_crystal',
    'neutral_monster', 'hook_aim', 'skill_effect'
]

current_class = 0
drawing = False
ix, iy = -1, -1
img_original = None
img_display = None
boxes = []
modified = False
scale = 1.0
img_path = None


def resize_to_fit(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        return resized, scale
    return img, 1.0


def draw_boxes(img, boxes, scale):
    for (x1, y1, x2, y2, cls_id) in boxes:
        x1_s = int(x1 * scale)
        y1_s = int(y1 * scale)
        x2_s = int(x2 * scale)
        y2_s = int(y2 * scale)
        cv2.rectangle(img, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
        cv2.putText(img, CLASSES[cls_id], (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_display, boxes, current_class, scale, img_original, modified
    if img_original is None:
        return
    x_raw = int(x / scale)
    y_raw = int(y / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_raw, y_raw
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_original.copy()
            img_display, _ = resize_to_fit(img_display)
            x1_s = int(min(ix, x_raw) * scale)
            y1_s = int(min(iy, y_raw) * scale)
            x2_s = int(max(ix, x_raw) * scale)
            y2_s = int(max(iy, y_raw) * scale)
            cv2.rectangle(img_display, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1 = min(ix, x_raw)
        y1 = min(iy, y_raw)
        x2 = max(ix, x_raw)
        y2 = max(iy, y_raw)
        if x2 - x1 > 5 and y2 - y1 > 5:
            boxes.append((x1, y1, x2, y2, current_class))
            modified = True
        img_display = img_original.copy()
        img_display, _ = resize_to_fit(img_display)
        draw_boxes(img_display, boxes, scale)


def load_yolo(img_shape, txt_path):
    """Load YOLO txt lines into a box list of (x1, y1, x2, y2, cls_id) pixel coords."""
    loaded = []
    if not txt_path.exists():
        return loaded
    h, w = img_shape[:2]
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            continue
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        loaded.append((x1, y1, x2, y2, cls_id))
    return loaded


def save_yolo(img_shape, boxes, txt_path):
    h, w = img_shape[:2]
    with open(txt_path, 'w') as f:
        for (x1, y1, x2, y2, cls_id) in boxes:
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def refresh_display():
    """Rebuild img_display from img_original and redraw all boxes."""
    global img_display, scale
    img_display = img_original.copy()
    img_display, _ = resize_to_fit(img_display)
    draw_boxes(img_display, boxes, scale)


def main():
    global img_original, img_display, boxes, current_class, scale, modified
    if len(sys.argv) < 2:
        print("Usage: python annotate.py <image_dir>")
        sys.exit(1)

    img_dir = Path(sys.argv[1])
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print("No images found.")
        return

    cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotation', 1280, 720)
    cv2.setMouseCallback('Annotation', mouse_callback)

    print("Controls: n(next), p(prev), c(change class), s(save), l(load), d(delete), q(quit)")
    print(f"Current class: {CLASSES[current_class]}")

    idx = 0
    while True:
        img_path = img_files[idx]
        img_original = cv2.imread(str(img_path))
        if img_original is None:
            idx = (idx + 1) % len(img_files)
            continue
        img_display, scale = resize_to_fit(img_original)
        txt_path = img_path.with_suffix('.txt')
        # Auto-load existing labels if present (merge target for future edits).
        boxes = load_yolo(img_original.shape, txt_path)
        modified = False
        draw_boxes(img_display, boxes, scale)
        if boxes:
            print(f"Loaded {len(boxes)} existing box(es) from {txt_path.name}")

        while True:
            cv2.imshow('Annotation', img_display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('n'):
                if modified and boxes:
                    save_yolo(img_original.shape, boxes, txt_path)
                    print(f"Saved to {txt_path}")
                idx = (idx + 1) % len(img_files)
                break
            elif key == ord('p'):
                if modified and boxes:
                    save_yolo(img_original.shape, boxes, txt_path)
                    print(f"Saved to {txt_path}")
                idx = (idx - 1) % len(img_files)
                break
            elif key == ord('c'):
                current_class = (current_class + 1) % len(CLASSES)
                print(f"Current class: {CLASSES[current_class]}")
            elif key == ord('s'):
                if boxes:
                    save_yolo(img_original.shape, boxes, txt_path)
                    modified = False
                    print(f"Saved to {txt_path}")
            elif key == ord('l'):
                boxes = load_yolo(img_original.shape, txt_path)
                modified = False
                refresh_display()
                print(f"Loaded {len(boxes)} box(es) from {txt_path}")
            elif key == ord('d'):
                if boxes:
                    boxes.pop()
                    modified = True
                    refresh_display()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
