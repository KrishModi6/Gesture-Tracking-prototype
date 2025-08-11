# === Hand Gesture Thing (Grade 10 vibes) ===
# Python 12 is not supported yet. Please use python 11
# Keys:
#   q = quit
#   r = start/stop recording samples
#   1..5 = which gesture you are recording
#   e = print stats
# https://www.python.org/downloads/release/python-3119/

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

# Auto-install dependencies
install_if_missing("cv2")  # Installed via opencv-python
install_if_missing("numpy")
install_if_missing("mediapipe")
import cv2
import time
import os
import csv
from collections import deque
import numpy as np
import mediapipe as mp

# --- mediapipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

GESTURES = ["open_palm", "fist", "peace_sign", "thumbs_up", "point"]

def open_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # sometimes Windows needs this
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

def make_dirs():
    base = "gesture_data"
    if not os.path.exists(base):
        os.makedirs(base)
    for g in GESTURES:
        p = os.path.join(base, g)
        if not os.path.exists(p):
            os.makedirs(p)
    # make a csv file header if it doesn't exist
    csv_path = os.path.join(base, "dataset.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # timestamp, gesture, then 21 landmarks x,y,z (63 numbers)
            header = ["timestamp", "gesture"] + [f"lm{i}_{c}" for i in range(21) for c in ["x","y","z"]]
            w.writerow(header)
    return base

def save_sample(base_folder, gesture, landmarks, frame):
    ts = int(time.time()*1000)
    # write image
    img_path = os.path.join(base_folder, gesture, f"{gesture}_{ts}.jpg")
    try:
        cv2.imwrite(img_path, frame)
    except Exception as e:
        print("couldn't save image:", e)

    # write csv
    row = [ts, gesture]
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    try:
        with open(os.path.join(base_folder, "dataset.csv"), "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print("couldn't write csv:", e)

def norm_dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def fingers_up_list(landmarks, handedness_label):
    # returns [thumb, index, middle, ring, pinky] as booleans
    tips = [4, 8, 12, 16, 20]
    mcps = [3, 5, 9, 13, 17]
    out = []

    # thumb: x direction flips for left vs right
    tip = landmarks[tips[0]]
    mcp = landmarks[mcps[0]]
    if handedness_label == "Right":
        out.append(tip.x > mcp.x)
    else:
        out.append(tip.x < mcp.x)

    # other fingers: tip higher (smaller y) than MCP
    for i in range(1,5):
        out.append(landmarks[tips[i]].y < landmarks[mcps[i]].y)
    return out

def guess_gesture(landmarks, handedness_label):
    # super simple rules (good enough for demo)
    f = fingers_up_list(landmarks, handedness_label)

    # (optional) normalized distances if you want to expand later
    scale = norm_dist(landmarks[0], landmarks[9]) + 1e-6
    thumb_index = norm_dist(landmarks[4], landmarks[8]) / scale
    index_middle = norm_dist(landmarks[8], landmarks[12]) / scale
    # ^ not really used right now, but helps if you add pinch/etc.

    if all(f):
        return "open_palm", 0.90
    if not any(f):
        return "fist", 0.95
    if f[1] and f[2] and (not f[3]) and (not f[4]):
        return "peace_sign", 0.85
    if f[0] and (not any(f[1:])):
        return "thumbs_up", 0.80
    if f[1] and (not any(f[2:])) and (not f[0]):
        return "point", 0.80
    return "unknown", 0.30

def main():
    print("Starting Hand Gesture Thing...")
    print("If it crashes, install stuff with:")
    print("  pip install opencv-python mediapipe numpy")
    print("Keys: q quit | r record | 1..5 pick gesture | e stats")

    base_folder = make_dirs()

    # state
    recording = False
    recording_label = None
    hist = deque(maxlen=8)  # smooth predictions a bit
    counts = {g:0 for g in GESTURES + ["unknown"]}
    times = []

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = open_cam()
    if not cap.isOpened():
        print("Camera not found :(")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        results = hands.process(rgb)
        dt = time.time() - t1
        times.append(dt)

        gesture_text = "No hand"
        conf = 0.0

        if results.multi_hand_landmarks:
            # zip landmarks with handedness so thumb logic works
            for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # "Left" or "Right"

                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                g, c = guess_gesture(hand_lms.landmark, label)
                hist.append(g)
                # pick most common in recent history
                gesture_text = max(set(hist), key=hist.count)
                # pretend confidence is average-ish
                conf = round(c, 2)

                # update counts
                counts[gesture_text] = counts.get(gesture_text, 0) + 1

                # save if recording
                if recording and recording_label in GESTURES:
                    save_sample(base_folder, recording_label, hand_lms.landmark, frame)

                # only use first hand for UI text to keep it simple
                break

        frames += 1
        elapsed = time.time() - t0
        fps = frames / elapsed if elapsed > 0 else 0

        # UI text
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), font, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, 60), font, 0.6, (255,0,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-140, 30), font, 0.6, (255,0,0), 2)

        cv2.putText(frame, "r=record 1..5=label  e=stats  q=quit", (10, h-15), font, 0.5, (0,0,255), 1)
        if recording:
            cv2.putText(frame, f"REC [{recording_label}]", (w-180, 60), font, 0.7, (0,0,255), 2)

        cv2.imshow("Hand Gesture Thing", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print("Recording:", "ON" if recording else "OFF")
        elif key == ord('e'):
            if times:
                avg = np.mean(times)*1000
                print("\n=== Stats ===")
                print(f"avg detect time: {avg:.2f} ms  (~{1/(np.mean(times)+1e-9):.1f} fps in pipeline)")
            total = sum(counts.values())
            print("detections:")
            for k,v in counts.items():
                pct = (v/total*100) if total else 0
                print(f"  {k:12s}: {v}  ({pct:.1f}%)")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            idx = int(chr(key)) - 1
            recording_label = GESTURES[idx]
            print("Recording label set to:", recording_label)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # helps some systems close the window

if __name__ == "__main__":
    main()
