import time
import cv2
from yolov4.tf import YOLOv4

def performYolov4(frame):
    predict_start_time = time.time()
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = yolo.predict(rgbImage, prob_thresh=0.25)
    frame = yolo.draw_bboxes(frame, bboxes)
    predict_exec_time = time.time() - predict_start_time
    cv2.putText(
        frame,
        "FPS - predict: {:.1f}".format(
            1 / predict_exec_time,
        ),
        org=(5, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(50, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return frame, bboxes

this_model = "coco_model"

yolo = YOLOv4()
yolo.config.parse_names("{}/obj.names".format(this_model))
yolo.config.parse_cfg("{}/yolov4-tiny-custom_d.cfg".format(this_model))

yolo.make_model()
yolo.load_weights("{}/yolov4-tiny-custom_last.weights".format(this_model), weights_type="yolo")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if ret:
        frame, bboxes = performYolov4(frame)
        cv2.imshow("dectection test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()