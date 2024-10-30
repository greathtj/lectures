import cv2
from djitellopy import Tello
import threading
import time

def show_video(drone:Tello):
    while True:
        # Get the video frame from the drone
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize the frame (optional)
        frame = cv2.resize(frame, (640, 480))

        # Display the frame
        cv2.imshow('Tello Video Feed', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()

v_thread = threading.Thread(target=show_video, args=(tello,))
v_thread.start()



# Takeoff
tello.takeoff()
time.sleep(5)

# Move up
tello.move_up(100)  # Move up 100 cm
time.sleep(2)

# Rotate clockwise
tello.rotate_clockwise(360)  # Rotate 90 degrees
time.sleep(2)

# Move forward
tello.rotate_counter_clockwise(360)  # Move forward 100 cm
time.sleep(2)

# Land
tello.land()



v_thread.join()
tello.streamoff()
cv2.destroyAllWindows()
