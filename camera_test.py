import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
print("摄像头已打开，按 Q 退出")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
