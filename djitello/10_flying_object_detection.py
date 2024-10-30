import cv2
from djitellopy import Tello
from ultralytics import YOLO
import threading
import time

def video_process(drone:Tello):
    global video_running

    while video_running:
        # Get the video frame from the drone
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize the frame (optional)
        frame = cv2.resize(frame, (640, 480))

        # object detection part
        results = model(frame, verbose=False)   # detection calculation
        frame = results[0].plot()               # draw the result on the image

        # Display the frame
        cv2.imshow('Tello Video Feed', frame)
        cv2.waitKey(1)


model = YOLO("yolo11n.pt")

# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()
time.sleep(2)

video_running = True
video_thread = threading.Thread(target=video_process, args=(tello,))
video_thread.start()

tello.takeoff()
time.sleep(1)
tello.move_up(50)
time.sleep(1)

tello.rotate_clockwise(360)
time.sleep(1)
tello.rotate_counter_clockwise(360)
time.sleep(1)

tello.land()
tello.streamoff()

video_running = False
video_thread.join()

cv2.destroyAllWindows()
