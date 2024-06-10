import os

from ultralytics import YOLO
import cv2

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # load a custom model

img = cv2.imread(r"images\corgi5.jpg")
img = cv2.resize(img, (600,600))

results = model(img)[0]
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)