import cv2
import mediapipe as mp
import numpy as np
import os
import time

# CHANGE THIS TO THE LETTER YOU ARE COLLECTING
LABEL = "c"   # change to B, C later
DATA_DIR = "../data"

SAVE_PATH = os.path.join(DATA_DIR, LABEL)
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
count = 0
MAX_SAMPLES = 200   # collect 200 samples per sign

print(f"Collecting data for sign: {LABEL}")
print("Press 's' to save sample | 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                np.save(os.path.join(SAVE_PATH, f"{count}.npy"), landmarks)
                count += 1
                print(f"Saved sample {count}")

    cv2.putText(
        frame,
        f"Samples: {count}/{MAX_SAMPLES}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collecting ISL Data", frame)

    if count >= MAX_SAMPLES or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
