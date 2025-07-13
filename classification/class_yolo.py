from ultralytics import YOLO
import cv2
import serial
import time

ARDUINO_PORT = 'COM4' # <<< CHANGE THIS TO YOUR ARDUINO'S PORT
BAUD_RATE = 9600 # Must match the baud rate in your Arduino sketch

ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

cap = cv2.VideoCapture(0)
model = YOLO("runs/classify/train2/weights/best.pt")

is_pumping = False

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, verbose=False)
        frame = results[0].plot()
        cv2.imshow("yolo classification", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("1"):
            ser.write("1\n".encode("utf-8"))
            is_pumping = True
        elif key == ord("0"):
            ser.write("0\n".encode("utf-8"))
            is_pumping = False

        names = results[0].names
        top1_index = results[0].probs.top1
        top1_conf = results[0].probs.top1conf.tolist()
        print(names, top1_index, top1_conf)

        if top1_index == 1 and top1_conf >= 0.98:
            if is_pumping:
                ser.write("0\n".encode("utf-8"))
            is_pumping = False

ser.close()
cap.release()
cv2.destroyAllWindows()