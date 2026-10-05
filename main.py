import cv2
import mediapipe as mp
import math
import pygame
import os
from ultralytics import YOLO
# Step 1: Pygame Mixer ko Sound ke liye initialize karna
pygame.mixer.init()
sound_path = "sounds/alarm.mp3"

# Check karna ki alarm file folder me hai ya nahi
if os.path.exists(sound_path):
    pygame.mixer.music.load(sound_path)
    print("🔔 Alarm Sound Successfully Load Ho Gaya!")
else:
    print("⚠️ Warning: 'sounds/alarm.mp3' nahi mili. Sound bajane ke liye folder me audio file dalein.")

# Step 2: MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils

# Load YOLOv8
model = YOLO("yolov8n.pt")

# ==========================
# Load YOLOv8 Model
# ==========================
model = YOLO("yolov8n.pt")

# Step 3: Math function do points ke beech ki doori (Distance) nikalne ke liye
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Step 4: EAR (Eye Aspect Ratio) nikalne ka formula
def get_ear(eye_points, landmarks):
    # Vertical points
    p2_p6 = calculate_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    p3_p5 = calculate_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal point
    p1_p4 = calculate_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    # EAR Formula
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

# Step 5: MAR (Mouth Aspect Ratio) nikalne ka formula ubaasi (Yawn) ke liye
def get_mar(mouth_points, landmarks):
    vertical = calculate_distance(landmarks[mouth_points[1]], landmarks[mouth_points[2]])
    horizontal = calculate_distance(landmarks[mouth_points[0]], landmarks[mouth_points[3]])
    mar = vertical / horizontal
    return mar

# MediaPipe ke fixed face point IDs (Indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308] # Left corner, Top lip, Bottom lip, Right corner

# 📈 AI/ML Threshold Configuration (Inhe train/tune kiya gaya hai)
EAR_THRESHOLD = 0.22   # Agar EAR isse kam hua toh aankh band mani jayegi
MAR_THRESHOLD = 0.50   # Agar MAR isse zyada hua toh muh khula (Yawn) mana jayega
EYE_FPS_LIMIT = 20     # Kitne frames tak aankh band rehne par alarm baje (Approx 1.5 - 2 Seconds)
YAWN_FPS_LIMIT = 25    # Kitne frames tak ubaasi chalne par warning baje

# Counters (Track rakhne ke liye)
EYE_COUNTER = 0
YAWN_COUNTER = 0
alarm_playing = False

# Camera Start
cap = cv2.VideoCapture(0)

print("🚀 Driver Monitoring System Active Hai... Band karne ke liye 'q' dabayein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Mirror Effect
    frame = cv2.flip(frame, 1)

    # =========================
    # YOLO Mobile Phone Detection
    # =========================
    yolo_results = model(frame, verbose=False)

    for result in yolo_results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            class_name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame,
                        class_name,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2)

            # Mobile Phone Detection
            if class_name == "cell phone":

                status_text = "PHONE DETECTED"

                status_color = (0,0,255)

                if not alarm_playing and os.path.exists(sound_path):
                    pygame.mixer.music.play(-1)
                    alarm_playing = True

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
            

    # Frame ko flip karna (Mirror effect ke liye) aur RGB me convert karna
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "Driver Active"
    status_color = (0, 255, 0) # Green Color

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Saare 468+ points ke coordinates (X, Y) pixel values me nikalna
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # EAR aur MAR calculate karna
            left_ear = get_ear(LEFT_EYE, landmarks)
            right_ear = get_ear(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            avg_mar = get_mar(MOUTH, landmarks)

            # --- LOGIC 1: DROWSINESS (NEEND) DETECTION ---
            if avg_ear < EAR_THRESHOLD:
                EYE_COUNTER += 1
                if EYE_COUNTER >= EYE_FPS_LIMIT:
                    status_text = "!!! DROWSINESS ALERT !!!"
                    status_color = (0, 0, 255) # Red Color
                    if not alarm_playing and os.path.exists(sound_path):
                        pygame.mixer.music.play(-1) # -1 ka matlab loop me bajta rahega
                        alarm_playing = True
            else:
                EYE_COUNTER = 0

            # --- LOGIC 2: YAWNING (UBAASI) DETECTION ---
            if avg_mar > MAR_THRESHOLD:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= YAWN_FPS_LIMIT:
                    # Agar banda so nahi raha par ubaasi le raha hai toh soft warning
                    if status_text == "Driver Active":
                        status_text = "Warning: Take a Break (Yawning)"
                        status_color = (0, 165, 255) # Orange Color
            else:
                YAWN_COUNTER = 0

            # --- ALARM STOP LOGIC ---
            # Agar driver ki aankhein khul gayi hain aur ubaasi bhi band hai, toh sound rok do
            if avg_ear >= EAR_THRESHOLD and alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

            # Screen par standard values print karna (Viva me dikhane ke liye)
            cv2.putText(frame, f"EAR (Eyes): {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR (Mouth): {avg_mar:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Big Status Text Display on Screen
    cv2.putText(frame, status_text, (30, 120), cv2.FONT_HERSHEY_COMPLEX, 0.8, status_color, 2)

    # Output Window Screen
    cv2.imshow('Driver Alertness System (Backend)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
