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
import pyautogui

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")

# Hand detection setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 60 FPS if your camera supports it

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange

# Control variables
current_mode = "IDLE"
last_gesture_time = time.time()
click_start_time = 0
is_clicking = False
is_dragging = False

# Mouse control rectangle position (bottom-right corner)
RECT_X_START = 350  # X-coordinate of the top-left corner of the rectangle
RECT_Y_START = 300  # Y-coordinate of the top-left corner of the rectangle
RECT_WIDTH = 120  # Width of the rectangle
RECT_HEIGHT = 90  # Height of the rectangle

# Mouse control variables
prev_x, prev_y = pyautogui.position()
smoothing = 2  # Reduced smoothing for faster response


def get_landmarks(frame, processed):
    landmarks_list = []
    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            for idx, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([idx, cx, cy])
            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
    return landmarks_list


def get_distance(p1, p2):
    return hypot(p1[1] - p2[1], p1[2] - p2[2])


def detect_gesture(landmarks):
    if not landmarks:
        return "NO_HAND"

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check for IDLE mode (pinky extended)
    if get_distance(landmarks[17], pinky_tip) > 50:
        return "IDLE"

    # Check for MOUSE mode (index finger extended)
    if (get_distance(thumb_tip, index_tip) > 10 and
            all(get_distance(ring_tip, tip) < 50 for tip in [middle_tip, pinky_tip])):
        return "MOUSE"

    # Check for CONTROL mode (index and middle fingers extended)
    if (get_distance(ring_tip, pinky_tip) < 50 and
            all(get_distance(pinky_tip, tip) > 5 for tip in [thumb_tip, index_tip, middle_tip])):
        return "CONTROL"

    return "OTHER"


def handle_mouse_control(landmarks, frame):
    global is_clicking, click_start_time, is_dragging, prev_x, prev_y

    if not landmarks:
        return

    # Use thumb tip position (landmark 4) for cursor control
    hand_mouse_x, hand_mouse_y = landmarks[0][1], landmarks[0][2]

    # Draw the rectangle on the frame (bottom-right corner)
    cv2.rectangle(frame, (RECT_X_START, RECT_Y_START),
                  (RECT_X_START + RECT_WIDTH, RECT_Y_START + RECT_HEIGHT),
                  (0, 255, 0), 2)

    # Map the thumb movement within the rectangle to the entire screen
    if RECT_X_START <= hand_mouse_x <= RECT_X_START + RECT_WIDTH and RECT_Y_START <= hand_mouse_y <= RECT_Y_START + RECT_HEIGHT:
        screen_width, screen_height = pyautogui.size()
        screen_x = np.interp(hand_mouse_x, [RECT_X_START, RECT_X_START + RECT_WIDTH], [0, screen_width])
        screen_y = np.interp(hand_mouse_y, [RECT_Y_START, RECT_Y_START + RECT_HEIGHT], [0, screen_height])

        # Apply smoothing to make the cursor movement smoother
        curr_x = prev_x + (screen_x - prev_x) / smoothing
        curr_y = prev_y + (screen_y - prev_y) / smoothing

        # Move the cursor
        pyautogui.moveTo(curr_x, curr_y)

        # Update previous position
        prev_x, prev_y = curr_x, curr_y

    # Check for click gestures using thumb and index finger
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    if get_distance(thumb_tip, index_tip) < 40:
        if not is_clicking:
            is_clicking = True
            click_start_time = time.time()
        elif time.time() - click_start_time > 0.5 and not is_dragging:
            # Start dragging
            pyautogui.mouseDown()
            is_dragging = True
    else:
        if is_clicking:
            if time.time() - click_start_time < 0.5:
                pyautogui.click()
            elif is_dragging:
                # Stop dragging
                pyautogui.mouseUp()
                is_dragging = False
            is_clicking = False


def handle_brightness_volume_control(landmarks):
    if not landmarks:
        return

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    length = get_distance(thumb_tip, index_tip)

    # Brightness control (left side of the screen)
    if thumb_tip[1] < 320:
        brightness = np.interp(length, [20, 200], [0, 100])
        sbc.set_brightness(brightness)
        socketio.emit('gesture_response', {'action': 'adjust_brightness', 'value': brightness})

    # Volume control (right side of the screen)
    else:
        vol = np.interp(length, [20, 200], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        socketio.emit('gesture_response', {'action': 'adjust_volume', 'value': vol})


def generate_frames():
    global current_mode, last_gesture_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmarks = get_landmarks(frame, processed)
        gesture = detect_gesture(landmarks)

        # Mode switching logic
        if time.time() - last_gesture_time > 1:  # Prevent rapid mode switches
            if gesture == "IDLE":
                current_mode = "IDLE"
                last_gesture_time = time.time()
            elif gesture == "MOUSE":
                current_mode = "MOUSE"
                last_gesture_time = time.time()
            elif gesture == "CONTROL":
                current_mode = "CONTROL"
                last_gesture_time = time.time()

        # Mode-specific actions
        if current_mode == "MOUSE":
            handle_mouse_control(landmarks, frame)
        elif current_mode == "CONTROL":
            handle_brightness_volume_control(landmarks)

        # Display current mode on frame
        cv2.putText(frame, f"Mode: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
