import cv2
from djitellopy import Tello

# Load the Haar Cascade for face detection
# down load the file from
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to track the face and move the drone accordingly
def track_face(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Connect to Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Turn on the video stream
tello.streamon()

while True:
    # Get the video frame from the drone
    frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize the frame (optional)
    frame = cv2.resize(frame, (640, 480))

    # Track the face and move the drone
    track_face(frame)

    # Display the frame
    cv2.imshow('Tello Video Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()