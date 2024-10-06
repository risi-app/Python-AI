from flask import Flask, Response
from flask_socketio import SocketIO
import cv2
import time
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from math import hypot

app = Flask(__name__)
socketio = SocketIO(app)

# Hand detection and volume/brightness control setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Setting up volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange

# Recognition delay timers
left_hand_detected_start = None
right_hand_detected_start = None
RECOGNITION_DELAY = 1.5  # 1.5 seconds delay

# Function to get landmarks
def get_left_right_landmarks(frame, processed):
    left_landmarks_list = []
    right_landmarks_list = []

    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            height, width, _ = frame.shape
            wrist_x = int(handlm.landmark[0].x * width)  # Get x-coordinate of the wrist (landmark 0)

            for idx, found_landmark in enumerate(handlm.landmark):
                x, y = int(found_landmark.x * width), int(found_landmark.y * height)

                if idx == 4 or idx == 8:
                    landmark = [idx, x, y]

                    # Classify hand based on wrist position
                    if wrist_x < width // 2:  # If wrist is on the left half of the screen
                        left_landmarks_list.append(landmark)
                    else:  # If wrist is on the right half of the screen
                        right_landmarks_list.append(landmark)

            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return left_landmarks_list, right_landmarks_list

# Function to calculate distance
def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return None

    (x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), (landmark_list[1][1], landmark_list[1][2])

    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    L = hypot(x2 - x1, y2 - y1)

    return L

# Function to stream video
def generate_frames():
    global left_hand_detected_start, right_hand_detected_start

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed)
            current_time = time.time()

            # Handle left hand for brightness
            if left_landmark_list:
                if left_hand_detected_start is None:
                    left_hand_detected_start = current_time
                elif current_time - left_hand_detected_start >= RECOGNITION_DELAY:
                    left_distance = get_distance(frame, left_landmark_list)
                    if left_distance:
                        b_level = np.interp(left_distance, [10, 200], [0, 100])
                        sbc.set_brightness(b_level)
                        socketio.emit('gesture_response', {'action': 'adjust_brightness', 'value': b_level})
            else:
                left_hand_detected_start = None

            # Handle right hand for volume
            if right_landmark_list:
                if right_hand_detected_start is None:
                    right_hand_detected_start = current_time
                elif current_time - right_hand_detected_start >= RECOGNITION_DELAY:
                    right_distance = get_distance(frame, right_landmark_list)
                    if right_distance:
                        vol = np.interp(right_distance, [10, 150], [minVol, maxVol])
                        volume.SetMasterVolumeLevel(vol, None)
                        socketio.emit('gesture_response', {'action': 'adjust_volume', 'value': vol})
            else:
                right_hand_detected_start = None

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)