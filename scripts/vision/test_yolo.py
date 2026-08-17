# scripts/vision/test_yolo.py
from ultralytics import YOLO
import cv2

# 加载预训练模型（首次运行会自动下载 yolov8n.pt）
model = YOLO("yolov8n.pt")

# 使用摄像头或图片测试，这里我们打开默认摄像头（USB手机画面可通过虚拟摄像头软件接入）
# 如果你已经用 scrcpy 将手机画面投屏到电脑，可以指定窗口名称或屏幕区域。
# 为了简单测试，我们用电脑摄像头：
cap = cv2.VideoCapture(0)  # 0 为默认摄像头，后续可改为手机画面源

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 执行检测
    results = model(frame)

    # 渲染结果
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 实时检测", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()