import cv2
import numpy as np
from insightface.app import FaceAnalysis
name = input("输入这个人的名字: ").strip()
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture(0)
print("请让这个人正对摄像头")
print("按 SPACE 拍摄，按 Q 退出")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.get(frame)
    cv2.imshow("Enroll", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == 32:
        if len(faces) != 1:
            print("必须检测到且只能检测到一个人脸")
            continue
        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)
        filename = f"{name}.npy"
        np.save(filename, embedding)
        print(f"已保存: {filename}")
        break
cap.release()
cv2.destroyAllWindows()
