import cv2
from djitellopy import Tello
from ultralytics import YOLO
import threading
import time

def video_process(drone:Tello):
    global video_running, key, objects

    while video_running:
        # Get the video frame from the drone
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize the frame (optional)
        frame = cv2.resize(frame, (640, 480))

        # object detection part
        results = model(frame, verbose=False)   # detection calculation
        frame = results[0].plot()               # draw the result on the image
        objects = results[0].boxes.cls.tolist()

        # Display the frame
        cv2.imshow('Tello Video Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break        


model = YOLO("best.pt")

# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()
time.sleep(2)

video_running = True
key = -1
objects = []
video_thread = threading.Thread(target=video_process, args=(tello,))
video_thread.start()

tello.takeoff()
time.sleep(1)
tello.move_up(50)
time.sleep(1)

# control by object detection
while True:
    if key == ord('q'):
        break
    elif len(objects)>0:
        if objects[0] == 0.0:
            tello.move_up(30)
        elif objects[0] == 1.0:
            tello.move_down(30)
    time.sleep(0.01)

tello.land()
tello.streamoff()

video_running = False
video_thread.join()

cv2.destroyAllWindows()
