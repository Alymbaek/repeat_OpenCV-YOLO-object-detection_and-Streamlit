import time

import cv2
import datetime
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
def path(paths):
    os.makedirs(paths, exist_ok=True)

path_image = 'yolov_images'
path(path_image)
path_videos = 'yolov_videos'
path(path_videos)

os.makedirs(path_videos)

if not path_image:
    os.makedirs(path_image)


if not cap.isOpened():
    print('Camera not opened')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_height = 30


counter = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print('Empty Capture')
        break

    predict = model(frame, stream=True)
    for i in predict:
        for n in i.boxes:
            x, y, w, h = map(int, n.xyxy[0])
            conf = float(n.conf[0])
            cls = int(n.cls[0])
            label = model.names[cls]

            if conf < 0.5:
                continue

            cv2.rectangle(frame, (x, y), (w, h), (0,255, 0), 2)
            cv2.putText(frame, f'Class: {label}, accuracy: {round(conf, 2)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255, 255), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 3)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        image_file = f'{path_image}/images_{date}.jpg'
        cv2.imwrite(image_file, frame)
        counter += 1
        print(f'Count images: {counter}')
        cv2.putText(frame, 'Saved', (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)
        cv2.imshow('S for take a photo', frame)
        out.write(frame)

    elif key == ord('v'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        file_name = f'{path_videos}/video_{date}.mp4'
        out = cv2.VideoWriter(file_name, fourcc, frame_fps, (frame_width, frame_height))


cap.release()
out.release()
cv2.destroyAllWindows()