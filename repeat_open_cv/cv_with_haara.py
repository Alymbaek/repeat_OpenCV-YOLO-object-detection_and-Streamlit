import cv2
import os
import datetime

def path(paths):
    os.makedirs(paths, exist_ok=True)

path_image = 'images'
path(path_image)

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera not opened')
    exit()

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Empty Capture')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_data = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in image_data:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1,  (0, 0, 255), 1)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        image_file = f'{path_image}/image_{date}.jpg'
        cv2.imwrite(image_file, frame)
        counter += 1
        print(f'Count images: {counter}')
        cv2.putText(frame, 'Saved', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow('S for take the photo', frame)
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()
