import cv2
import time
import os
from datetime import datetime

# Set the desired resolution
TARGET_WIDTH = 800
TARGET_HEIGHT = 600

# Define the folder to save images
SAVE_FOLDER = "C:/Users/greathtj/Desktop/photos"

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Created folder: {SAVE_FOLDER}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set resolution for the camera feed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

continuous_shot_mode = False
last_shot_time = 0
shot_interval = 1 # seconds

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("My Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # Exit on 'q'
        if key == ord('q'):
            break

        # Get current timestamp for filename
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # YYYYMMDD_HHMMSS_milliseconds

        # Take a single shot on 's'
        if key == ord('s'):
            filename = os.path.join(SAVE_FOLDER, f"myshot_{current_time_str}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Single shot saved: {filename}")

        # Toggle continuous shot mode on 'c'
        if key == ord('c'):
            continuous_shot_mode = not continuous_shot_mode
            if continuous_shot_mode:
                print("Continuous shot mode: ON")
            else:
                print("Continuous shot mode: OFF")

        # Continuous shot logic
        if continuous_shot_mode:
            current_time = time.time()
            if current_time - last_shot_time >= shot_interval:
                filename = os.path.join(SAVE_FOLDER, f"continuous_shot_{current_time_str}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Continuous shot saved: {filename}")
                last_shot_time = current_time

cap.release()
cv2.destroyAllWindows()