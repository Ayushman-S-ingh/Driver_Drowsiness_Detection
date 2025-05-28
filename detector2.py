from flask import Flask, render_template, Response, jsonify
import cv2
import time
from threading import Thread
import pygame
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Global status variable to share between Flask and detection logic
current_status = "Normal"
eye_closed_start = None
ALARM_ON = False

# Initialize alarm sound
pygame.mixer.init()
pygame.mixer.music.load("static/alarm.mp3")

# Initialize MediaPipe FaceMesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Eye Aspect Ratio to detect drowsiness
def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    eye = [landmarks[i] for i in eye_indices]
    eye_points = [(int(p.x * image_width), int(p.y * image_height)) for p in eye]

    vertical = abs(eye_points[1][1] - eye_points[5][1]) + abs(eye_points[2][1] - eye_points[4][1])
    horizontal = abs(eye_points[0][0] - eye_points[3][0])

    if horizontal == 0:
        return 0

    return vertical / (2.0 * horizontal)

# Try to initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    cap = cv2.VideoCapture(1)  # Try second camera (index 1)
    if not cap.isOpened():
        print("Error: Could not access any camera.")
        exit()

COUNTER = 0

def sound_alarm():
    pygame.mixer.music.play(-1)  # Loop the alarm sound indefinitely

def stop_alarm():
    pygame.mixer.music.stop()  # Stop the alarm sound

def generate_frames():
    global COUNTER, ALARM_ON, current_status, eye_closed_start

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture image from camera.")
            break
        
        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Check if the eyes are closed for too long (drowsiness detection)
                if ear < 0.21:
                    COUNTER += 1
                    if COUNTER >= 15:  # Alarm will trigger after the eyes are closed for more than 2 seconds
                        if not ALARM_ON:
                            ALARM_ON = True
                            Thread(target=sound_alarm).start()
                        current_status = "Drowsy"
                else:
                    # Reset counter if eyes are open (normal mode)
                    if COUNTER > 0:  
                        COUNTER = 0
                    # Stop alarm when in normal mode
                    if ALARM_ON:
                        ALARM_ON = False
                        stop_alarm()
                    current_status = "Normal"

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(status=current_status)

if __name__ == "__main__":
    app.run(debug=True)
