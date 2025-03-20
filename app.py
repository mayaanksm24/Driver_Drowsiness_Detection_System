from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.initializers import Orthogonal
import threading
import time
import os
from pygame import mixer

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
alarm_status = False
alarm_status_buzz = False
detection_running = False
detection_thread = None

# Initialize pygame mixer for sound
mixer.init()
alarm_sound = "data/alarm.mp3"

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

# Load model
import keras
keras.utils.get_custom_objects()["Orthogonal"] = Orthogonal()
model = load_model("best_model_CNN_RNN.h5")
classes = ['Closed', 'Open']

def detect_drowsiness():
    global camera, output_frame, lock, alarm_status, alarm_status_buzz, detection_running
    
    print("Detection thread started")
    count = 0
    
    # Initialize camera inside the thread
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Failed to open camera in thread")
        detection_running = False
        return
    
    print("Camera opened successfully in thread")
    
    while detection_running:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        # Process the frame
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status1 = ''
        status2 = ''
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)
            
            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                if eye1.size == 0:
                    continue
                eye1 = cv2.resize(eye1, (145, 145))
                eye1 = eye1.astype('float') / 255.0
                eye1 = img_to_array(eye1)
                eye1 = np.expand_dims(eye1, axis=0)
                pred1 = model.predict(eye1)
                status1 = np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                if eye2.size == 0:
                    continue
                eye2 = cv2.resize(eye2, (145, 145))
                eye2 = eye2.astype('float') / 255.0
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                pred2 = model.predict(eye2)
                status2 = np.argmax(pred2)
                break

            # If the eyes are closed, start counting
            if status1 == 2 and status2 == 2:
                count += 1
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                # if eyes are closed for 10 consecutive frames, start the alarm
                if count >= 10:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    alarm_status = True
                    if not alarm_status_buzz:
                        alarm_status_buzz = True
                        # Play the alarm sound
                        try:
                            if not mixer.music.get_busy():
                                mixer.music.load(alarm_sound)
                                mixer.music.play(-1)  # Play in a loop
                        except Exception as e:
                            print(f"Error playing sound: {e}")
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_status = False
                if alarm_status_buzz:
                    alarm_status_buzz = False
                    # Stop the alarm sound
                    try:
                        mixer.music.stop()
                    except Exception as e:
                        print(f"Error stopping sound: {e}")
                
        # Display a message if no faces are detected
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                
        # Encode the frame in JPEG format
        with lock:
            output_frame = frame.copy()
    
    # Clean up when the thread ends
    if camera is not None:
        camera.release()
        print("Camera released in thread")
    
    print("Detection thread ended")

def generate():
    global output_frame, lock, detection_running
    
    while True:
        # Create a blank frame with status message
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        if not detection_running:
            cv2.putText(blank_frame, "Click 'Start Detection' to begin", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Return blank frame when not running
            (flag, encoded_image) = cv2.imencode(".jpg", blank_frame)
            if flag:
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encoded_image) + b'\r\n')
            time.sleep(0.1)
            continue
        
        with lock:
            if output_frame is None:
                # Return a "starting camera" message
                cv2.putText(blank_frame, "Starting camera...", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                (flag, encoded_image) = cv2.imencode(".jpg", blank_frame)
                if flag:
                    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                          bytearray(encoded_image) + b'\r\n')
                time.sleep(0.1)
                continue
            
            # We have a valid frame, so return it
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
            
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # Return the response generated along with the specific media type (MIME type)
    return Response(generate(),
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "alarm_status": alarm_status,
        "detection_running": detection_running
    })

@app.route("/start", methods=["POST"])
def start_detection():
    global detection_running, detection_thread, camera, output_frame
    
    if detection_running:
        return jsonify({"success": True, "message": "Detection already running"})
    
    # Reset the output frame
    with lock:
        output_frame = None
    
    # Set the flag to running
    detection_running = True
    
    # Start a new detection thread
    detection_thread = threading.Thread(target=detect_drowsiness)
    detection_thread.daemon = True
    detection_thread.start()
    
    return jsonify({"success": True, "message": "Detection started"})

@app.route("/stop", methods=["POST"])
def stop_detection():
    global detection_running, camera, output_frame, alarm_status, alarm_status_buzz
    
    if not detection_running:
        return jsonify({"success": True, "message": "Detection already stopped"})
    
    # Stop the detection thread
    detection_running = False
    
    # Stop alarm if it's playing
    if alarm_status_buzz:
        alarm_status_buzz = False
        alarm_status = False
        try:
            mixer.music.stop()
        except Exception as e:
            print(f"Error stopping sound: {e}")
    
    # Clear the output frame
    with lock:
        output_frame = None
    
    return jsonify({"success": True, "message": "Detection stopped"})

if __name__ == "__main__":
    # Start Flask app
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)