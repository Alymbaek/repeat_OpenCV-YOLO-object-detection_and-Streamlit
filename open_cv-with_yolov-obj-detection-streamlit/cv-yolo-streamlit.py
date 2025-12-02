import cv2
import streamlit as st
from ultralytics import YOLO
import time

st.title('OpenCV with Streamlit')
st.sidebar.header('settings')
choosed = st.sidebar.selectbox('Choose one model', ['yolov8n.pt', 'yolov9n.pt'])
conf = st.sidebar.slider('Min accuracy', 0.25, 0.9, 0.5, 0.05)
button_start = st.sidebar.button('Start')
button_stop = st.sidebar.button('Stop')

model = YOLO(choosed)


print_image = st.image([])

if button_start:
    st.success('Start model work')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print('Emtpy Capture')
            break
        if button_stop:
            break
        predict = model(frame, stream=True, conf=0.5)
        for i in predict:
            for n in i.boxes:
                conf = float(n.conf[0])
                cls = int(n.cls[0])
                label = model.names[cls]

                x, y, w, h = map(int, n.xyxy[0])
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)
                cv2.putText(frame, f'Class: {label}, accuracy: {round(conf, 2)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {round(fps,2 )}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print_image.image(rgb_image)

    cap.release()
    st.info('Model stop work')
