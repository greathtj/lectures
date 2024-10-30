import cv2
from djitellopy import Tello

# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()

# Takeoff
tello.takeoff()

while True:
    # Get the video frame from the drone
    frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize the frame (optional)
    frame = cv2.resize(frame, (640, 480))

    # Display the frame
    cv2.imshow('Tello Video Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Land
tello.land()

tello.streamoff()
cv2.destroyAllWindows()
