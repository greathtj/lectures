
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short range, 1 for full range
    min_detection_confidence=0.5
)

# Initialize MediaPipe Face Mesh (for more detailed facial landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start video capture
cap = cv2.VideoCapture(1)

# Get the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize FPS calculation variables
prev_time = 0
curr_time = 0

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face detection
        face_results = face_detection.process(rgb_frame)
        
        # Process the frame for face mesh
        mesh_results = face_mesh.process(rgb_frame)
        
        # Draw face detection results
        if face_results.detections:
            for detection in face_results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height), \
                       int(bboxC.width * frame_width), int(bboxC.height * frame_height)
                
                # Draw bounding box
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                
                # Draw face detection confidence
                confidence = detection.score[0]
                cv2.putText(frame, f'Confidence: {confidence:.2f}',
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw face mesh landmarks
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw the face mesh with custom settings
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {int(fps)}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display mode information
        cv2.putText(frame, "Press 'm' to toggle face mesh",
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle face mesh visibility (you can add this feature)
            pass

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
