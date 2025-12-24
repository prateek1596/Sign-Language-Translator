import cv2
import mediapipe as mp
import csv
import os

# ---------- NORMALIZATION FUNCTION ----------
def get_normalized_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    max_val = max([abs(v) for point in coords for v in point])

    normalized = []
    for point in coords:
        normalized.extend([v / max_val for v in point])

    return normalized

# ---------- SETUP ----------
label = input("Enter letter (A-Z): ").upper()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

file_exists = os.path.isfile("landmarks.csv")

with open("landmarks.csv", "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        writer.writerow(header)

    print("Hold the sign steady and press S to save | ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = get_normalized_landmarks(hand_landmarks)

                cv2.putText(frame, f"Letter: {label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and result.multi_hand_landmarks:
            data.append(label)
            writer.writerow(data)
            print("Saved")

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
