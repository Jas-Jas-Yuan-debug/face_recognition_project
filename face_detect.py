import cv2
import insightface
print("正在加载 InsightFace 模型，第一次可能需要下载模型...")
app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
print("模型加载完成！")
print("摄像头启动，按 Q 退出。")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break
    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Face {face.det_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
