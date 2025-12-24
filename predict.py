import cv2
import mediapipe as mp
import pickle
import pandas as pd
from collections import deque, Counter
import pyttsx3
import time

# ------------------ LOAD MODEL ------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

columns = model.feature_names_in_

# ------------------ TEXT TO SPEECH ------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ------------------ MEDIAPIPE ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ------------------ SMOOTHING ------------------
predictions = deque(maxlen=15)

# ------------------ WORD LOGIC ------------------
current_word = ""
last_added_letter = ""
last_hand_time = time.time()
NO_HAND_TIMEOUT = 1.5  # space after no hand

# ------------------ STABILITY CONTROL ------------------
STABLE_TIME = 0.8  # seconds sign must be held
last_pred = ""
pred_start_time = time.time()

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        last_hand_time = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            X_input = pd.DataFrame([data], columns=columns)
            pred = model.predict(X_input)[0]

            predictions.append(pred)
            final_pred = Counter(predictions).most_common(1)[0][0]

            # ---------- STABILITY FILTER ----------
            if final_pred != last_pred:
                last_pred = final_pred
                pred_start_time = time.time()
            else:
                if time.time() - pred_start_time > STABLE_TIME:
                    if final_pred != last_added_letter:
                        current_word += final_pred
                        last_added_letter = final_pred
                        pred_start_time = time.time()

            # Display current letter
            cv2.putText(frame, f"Letter: {final_pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            cv2.putText(frame, "Hold sign...", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ---------- SPACE (NO HAND) ----------
    if not hand_detected and time.time() - last_hand_time > NO_HAND_TIMEOUT:
        if current_word and not current_word.endswith(" "):
            current_word += " "
            last_added_letter = ""
            time.sleep(0.3)

    # ---------- DISPLAY WORD ----------
    cv2.putText(frame, f"Word: {current_word}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    # ENTER → Speak
    if key == 13 and current_word.strip():
        engine.say(current_word)
        engine.runAndWait()

    # BACKSPACE → delete
    elif key == 8:
        current_word = current_word[:-1]
        last_added_letter = ""

    # ESC → Exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
