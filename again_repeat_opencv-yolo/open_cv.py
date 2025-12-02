import cv2
import os
import datetime
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def path(path):
    os.makedirs(path, exist_ok=True)

path_images = 'images'
path(path_images)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Камера ачылган жок')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Камера пустой')
        break

    predict = model(frame, stream=True, conf=0.5)
    for i in predict:
        for n in i.boxes:
            cls = int(n.cls[0])
            conf = float(n.conf[0])
            label = model.names[cls]

            x, y, w, h = map(int, n.xyxy[0])
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(frame, f'class: {label}, accuracy: {round(conf, 2)}%', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        image_file = f'{path_images}/image_{date}.jpg'
        cv2.imwrite(image_file, frame)
        cv2.putText(frame, 'Суротко алынды', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.imshow('Screen', frame)


cap.release()
cv2.destroyAllWindows()


