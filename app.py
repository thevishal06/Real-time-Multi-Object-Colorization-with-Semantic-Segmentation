
import cv2
import numpy as np
from flask import Flask, render_template, Response
from models.segmentation_model import colorize_frame_with_segmentation

app = Flask(__name__)

# Video capture (replace 0 with video file path for a file-based stream)
video_capture = cv2.VideoCapture(0)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Function to process and stream video
def process_video_stream():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Perform real-time segmentation and colorization
        colorized_frame = colorize_frame_with_segmentation(frame)

        # Convert the colorized frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', colorized_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
