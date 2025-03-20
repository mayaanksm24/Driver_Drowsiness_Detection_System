Overview
This project implements a real-time Drowsiness Detection System using Flask, OpenCV, and Deep Learning. The system detects when a user's eyes are closed for a prolonged period and triggers an alarm to prevent accidents caused by drowsiness.

Features
Real-time eye detection using OpenCV
Deep learning-based eye status classification (Open/Closed)
Live video feed processing
Alarm system activation when drowsiness is detected
Flask-based web interface for easy interaction

Requirements
Ensure you have the following dependencies installed:
  pip install flask opencv-python numpy keras pygame
  
Installation & Usage
Clone the Repository
  git clone https://github.com/your-username/drowsiness-detection.git
  cd drowsiness-detection
  
Download Pre-trained Model

Place the best_model_CNN_RNN.h5 file inside the project directory.

Run the Application
  python app.py
The Flask server will start, and you can access the web interface at:
  http://127.0.0.1:5000/
  
API Endpoints
/ → Main web interface
/video_feed → Provides real-time video stream
/status → Returns detection status in JSON format
/start → Starts drowsiness detection
/stop → Stops drowsiness detection
