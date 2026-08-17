# scripts/vision/test_scrcpy.py
import cv2
from scrcpy_client import ScrcpyClient

# 创建客户端并连接
client = ScrcpyClient()
client.start()

print("已连接手机，按 'q' 退出画面显示...")

while True:
    # 获取最新帧（numpy array，BGR格式）
    frame = client.last_frame
    if frame is not None:
        cv2.imshow("手机实时画面", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.stop()
cv2.destroyAllWindows()