import cv2
import numpy as np

def do_object_detection(frame, names_file):
    # read names file
    with open(names_file, 'r') as nf:
        names = nf.readlines()

    # Perform object detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detected objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = names[class_ids[i]].strip()
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (0, 255, 0), 2)

    return frame

net = cv2.dnn.readNetFromDarknet("coco_model/yolov4-tiny-custom.cfg", "coco_model/yolov4-tiny-custom_last.weights")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()
    frame = do_object_detection(frame, "coco_model/obj.names")

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
