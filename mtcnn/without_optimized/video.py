import cv2
from src import detect_faces, show_bboxes
from PIL import Image
import numpy as np

capture = cv2.VideoCapture(0)

while (True):
    ret, frame = capture.read()     # type(frame): ndarray
    img = Image.fromarray(frame, mode='RGB')
    bounding_boxes, landmarks = detect_faces(img)
    image_2 = show_bboxes(img, bounding_boxes, landmarks)
    frame = np.array(image_2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# 优化的方法：直接传入ndararry给detect_faces()