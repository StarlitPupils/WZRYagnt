import cv2
import numpy as np

def nothing(x):
    pass

video_path = "data/raw_videos/type_a_ui_animation.mp4"
cap = cv2.VideoCapture(video_path)
# 跳转到箭头明显的帧（例如第500帧）
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
if not ret:
    print("读取失败")
    exit()

cv2.namedWindow('HSV Tuner')
cv2.createTrackbar('H Min', 'HSV Tuner', 0, 180, nothing)
cv2.createTrackbar('H Max', 'HSV Tuner', 180, 180, nothing)
cv2.createTrackbar('S Min', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Tuner', 30, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Tuner', 200, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('Area Min', 'HSV Tuner', 50, 500, nothing)
cv2.createTrackbar('Area Max', 'HSV Tuner', 2000, 5000, nothing)
cv2.createTrackbar('Aspect Min', 'HSV Tuner', 20, 100, nothing)  # 实际值=track/10

while True:
    h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')
    area_min = cv2.getTrackbarPos('Area Min', 'HSV Tuner')
    area_max = cv2.getTrackbarPos('Area Max', 'HSV Tuner')
    aspect_min = cv2.getTrackbarPos('Aspect Min', 'HSV Tuner') / 10.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w < 5 or h < 5:
            continue
        aspect = max(w,h) / (min(w,h)+1e-5)
        if aspect < aspect_min:
            continue
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(display, [box], 0, (0,255,0), 2)

    cv2.imshow('Mask', mask)
    cv2.imshow('HSV Tuner', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"推荐参数: H({h_min},{h_max}) S({s_min},{s_max}) V({v_min},{v_max}) Area({area_min},{area_max}) AspectMin({aspect_min:.1f})")
