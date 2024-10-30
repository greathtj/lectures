import cv2
import numpy as np
import mediapipe as mp
from djitellopy import Tello
import time
import threading

class TelloFaceTracker:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )

        # Initialize Tello drone
        self.tello = Tello()
        
        # Control parameters
        self.tracking_enabled = False
        self.speed = 30  # Speed of drone movements (1-100)
        self.center_deadzone = 100  # Pixels from center where no movement is needed
        self.frame = None
        self.battery_low = False
        self.is_flying = False

    def connect_drone(self):
        """Safely connect to the Tello drone"""
        try:
            self.tello.connect()
            print(f"Battery Level: {self.tello.get_battery()}%")
            return True
        except Exception as e:
            print(f"Error connecting to Tello: {e}")
            return False

    def initialize_stream(self):
        """Start the video stream"""
        try:
            self.tello.streamon()
            return True
        except Exception as e:
            print(f"Error starting video stream: {e}")
            return False

    def battery_monitor(self):
        """Monitor battery level in a separate thread"""
        while True:
            try:
                battery = self.tello.get_battery()
                self.battery_low = battery < 20
                if self.battery_low:
                    print(f"WARNING: Battery Low ({battery}%)")
                time.sleep(30)  # Check battery every 30 seconds
            except:
                pass

    def process_frame(self, frame):
        """Process frame for face detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate face center
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Draw face center and frame center
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
                
                # Draw deadzone
                cv2.rectangle(frame, 
                            (frame_center_x - self.center_deadzone, 
                             frame_center_y - self.center_deadzone),
                            (frame_center_x + self.center_deadzone, 
                             frame_center_y + self.center_deadzone),
                            (255, 0, 0), 2)
                
                if self.tracking_enabled and self.is_flying:
                    self.track_face(face_center_x, face_center_y, 
                                  frame_center_x, frame_center_y)
                
                # Display tracking status
                status = "Tracking: ON" if self.tracking_enabled else "Tracking: OFF"
                cv2.putText(frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        return frame

    def track_face(self, face_x, face_y, center_x, center_y):
        """Control drone based on face position"""
        x_diff = face_x - center_x
        y_diff = face_y - center_y
        
        # Only move if face is outside deadzone
        if abs(x_diff) > self.center_deadzone or abs(y_diff) > self.center_deadzone:
            # Left-Right movement
            if x_diff > self.center_deadzone:
                self.tello.send_rc_control(self.speed, 0, 0, 0)
            elif x_diff < -self.center_deadzone:
                self.tello.send_rc_control(-self.speed, 0, 0, 0)
                
            # Up-Down movement
            if y_diff > self.center_deadzone:
                self.tello.send_rc_control(0, 0, -self.speed, 0)
            elif y_diff < -self.center_deadzone:
                self.tello.send_rc_control(0, 0, self.speed, 0)
        else:
            # Hover in place if face is in deadzone
            self.tello.send_rc_control(0, 0, 0, 0)

    def run(self):
        """Main run loop"""
        if not self.connect_drone():
            return
        
        if not self.initialize_stream():
            return
        
        # Start battery monitoring in separate thread
        battery_thread = threading.Thread(target=self.battery_monitor, daemon=True)
        battery_thread.start()
        
        print("\nDrone Control Instructions:")
        print("T - Toggle tracking")
        print("L - Take off")
        print("U - Land")
        print("Q - Quit")
        print("\nSafety Features:")
        print("- Automatic landing on low battery")
        print("- Emergency land with Spacebar")
        print("- Tracking disabled until takeoff")
        
        try:
            while True:
                # Get frame from Tello
                frame = self.tello.get_frame_read().frame
                if frame is None:
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Show battery warning
                if self.battery_low:
                    cv2.putText(processed_frame, "LOW BATTERY!", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)
                
                # Display frame
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Tello Face Tracking", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.tracking_enabled = not self.tracking_enabled
                    print(f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
                elif key == ord('l'):
                    if not self.is_flying and not self.battery_low:
                        self.tello.takeoff()
                        self.is_flying = True
                elif key == ord('u') or key == 32:  # 'u' or spacebar
                    if self.is_flying:
                        self.tello.land()
                        self.is_flying = False
                
                # Safety check - land if battery is low
                if self.battery_low and self.is_flying:
                    print("Emergency landing due to low battery!")
                    self.tello.land()
                    self.is_flying = False
                    
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.tello.streamoff()
            if self.is_flying:
                self.tello.land()
            self.tello.end()

if __name__ == "__main__":
    tracker = TelloFaceTracker()
    tracker.run()