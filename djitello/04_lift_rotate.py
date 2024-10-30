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

# move up for safety
tello.move_up(50)
time.sleep(3)

# rotate clockwise
tello.rotate_clockwise(360)
time.sleep(3)

# rotate counter clockwise
tello.rotate_counter_clockwise(360)
time.sleep(3)

# Land
tello.land()
