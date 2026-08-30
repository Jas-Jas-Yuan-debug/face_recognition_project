import cv2
import numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture(0)
print("请正对摄像头，按 Q 保存你的人脸")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(
            frame,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            2
        )
    cv2.imshow("Register", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        if len(faces) == 1:
            embedding = faces[0].normed_embedding
            np.save("my_face.npy", embedding)
            print("已保存你的人脸")
            break
        else:
            print(f"检测到 {len(faces)} 张脸，请确保只有你一个人")
cap.release()
cv2.destroyAllWindows()
