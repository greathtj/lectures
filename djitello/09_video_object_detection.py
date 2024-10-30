import cv2
from djitellopy import Tello
from ultralytics import YOLO
import time

model = YOLO("yolo11n.pt")

# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()
time.sleep(2)

while True:
    # Get the video frame from the drone
    frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize the frame (optional)
    frame = cv2.resize(frame, (640, 480))

    # object detection part
    results = model(frame, verbose=False)   # detection calculation
    frame = results[0].plot()               # draw the result on the image

    # Display the frame
    cv2.imshow('Tello Video Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
