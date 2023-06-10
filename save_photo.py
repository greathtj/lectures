import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("my frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("myshot.jpg", frame)

cap.release()
cv2.destroyAllWindows()