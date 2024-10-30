from djitellopy import Tello
import time

# Create a Tello object
tello = Tello()

# Connect to the drone
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Takeoff
tello.takeoff()
time.sleep(5)

# Land
tello.land()
