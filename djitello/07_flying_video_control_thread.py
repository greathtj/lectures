import cv2
from djitellopy import Tello
import threading
import time

# ------------- Thread routine --------------------------------
def show_video(drone:Tello):
    global key

    while True:
        # Get the video frame from the drone
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize the frame (optional)
        frame = cv2.resize(frame, (640, 480))

        # Display the frame
        cv2.imshow('Tello Video Feed', frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# =============== main program ================================= 

# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()

key = -1
v_thread = threading.Thread(target=show_video, args=(tello,))
v_thread.start()



# Takeoff
tello.takeoff()
time.sleep(2)

# Move up
tello.move_up(50)  # Move up 50 cm
time.sleep(2)

while True:
    if key == ord('q'):
        break
    elif key == ord('i'):
        tello.move_up(30)
    elif key == ord('k'):
        tello.move_down(30)
    elif key == ord('j'):
        tello.move_left(30)
    elif key == ord('l'):
        tello.move_right(30)

    time.sleep(0.01)

# Land
tello.land()



v_thread.join()
tello.streamoff()
cv2.destroyAllWindows()
