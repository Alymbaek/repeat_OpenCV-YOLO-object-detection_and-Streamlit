# OpenCV + YOLO Practice Project

This repository contains my practice work with **OpenCV**, **YOLOv8**, **YOLOv9**, **Haar Cascade**, and **Streamlit**.  
The main goal of this project is simple: **learn computer vision by writing the code myself, without copying ready solutions**.

I repeated the logic many times, rewrote parts of the code from memory, experimented with the camera, saved images/videos, and built a small Streamlit interface.  
Some parts of the code may not be perfect yet — and that's totally fine. I am learning every day and improving step by step.

---

## 🔥 Features

### ✔ Real-time object detection using YOLO (Ultralytics)
- Live camera feed  
- Bounding boxes + class names  
- Confidence score  
- FPS counter  
- Save frames (`s`)
- Record video (`v`)

### ✔ Real-time face detection using Haar Cascades
- Face detection using classical CV  
- Save frames on key press  
- Simple and fast

### ✔ Streamlit Web App
- Select YOLO model (e.g. **yolov8n**, **yolov9n**)  
- Adjustable confidence slider  
- Start/Stop capture  
- Display annotated frames in the browser  



## 📁 Project Structure
repear_OpenCV-Yolo-Detection

again_repeat_opencv-yolo
     open_cv.py
     yolov8n.pt
     path_repeat_images

open_cv-with_yolov-obj-detection
     cv_with_yolo.py
     cv-yolo-streamlit.py
     yolov8n.pt
     yolov_images
     yolov_videos

reapear_open_cv
     cv_with_haara.py
     images
     main.py






## 🚀 How to Run

### 1. Create virtual environment

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
2. Install dependencies

pip install -r requirements.txt
3. Run any script you want:
YOLO + OpenCV:


python cv_with_yolo.py
Haar Cascade:


python cv_with_haara.py
Streamlit UI:



streamlit run cv-yolo-streamlit.py
## 📦 Requirements

opencv-python
ultralytics
streamlit
numpy
(You can add versions later when you freeze the environment.)

## 🎯 Learning Purpose
This project is not a production application.
It’s a learning project, where I:

repeat OpenCV basics many times

practice YOLO integration

build UI using Streamlit

debug camera issues

write code fully on my own to understand the logic

fix mistakes and learn from them

I know that some parts of the code are not perfect yet
But every file here was written by me manually — no copy-paste, no templates.
My goal is to improve small parts every day and become better in computer vision and Python backend development.

## 🧠 My Future Plans
Add GPU support

Improve folder structure

Rewrite Streamlit app using threads / session state

Add FastAPI backend for YOLO model

Add Docker support

Train a custom YOLO model and integrate it

## 💬 Final Words
This repository shows my real learning path — with mistakes, experiments, tests, and progress.
I’m learning consistently and working hard to master computer vision and AI step by step.
