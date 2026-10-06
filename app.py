import cv2
import time
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. LOAD YOUR TRAINED MODEL
# ============================================================

MODEL_PATH = "human_activity_model_prediction01.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model classes:", model.classes_)
print("Number of features:", model.n_features_in_)


# ============================================================
# 2. MEDIAPIPE POSE LANDMARKER
# ============================================================

POSE_MODEL_PATH = "pose_landmarker_full.task"

base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# ============================================================
# 3. FEATURE COLUMNS
#    EXACTLY MATCHES YOUR .PKL MODEL
# ============================================================

FEATURE_COLUMNS = [
    "Unnamed: 0",

    "nose_x",
    "nose_y",

    "left_shoulder_x",
    "left_shoulder_y",

    "right_shoulder_x",
    "right_shoulder_y",

    "left_elbow_x",
    "left_elbow_y",

    "right_elbow_x",
    "right_elbow_y",

    "left_wrist_x",
    "left_wrist_y",

    "right_wrist_x",
    "right_wrist_y",

    "left_hip_x",
    "left_hip_y",

    "right_hip_x",
    "right_hip_y",

    "left_knee_x",
    "left_knee_y",

    "right_knee_x",
    "right_knee_y",

    "left_ankle_x",
    "left_ankle_y",

    "right_ankle_x",
    "right_ankle_y",

    "left_heel_x",
    "left_heel_y",

    "right_heel_x",
    "right_heel_y",

    "left_foot_index_x",
    "left_foot_index_y",

    "right_foot_index_x",
    "right_foot_index_y",

    "left_eye_x",
    "left_eye_y"
]


# ============================================================
# 4. MEDIAPIPE LANDMARK INDEXES
# ============================================================

LANDMARKS = {
    "nose": 0,

    "left_eye": 2,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28,

    "left_heel": 29,
    "right_heel": 30,

    "left_foot_index": 31,
    "right_foot_index": 32
}


# ============================================================
# 5. FUNCTION TO EXTRACT 37 FEATURES
# ============================================================

def extract_features(pose_landmarks, frame_number):

    row = {}

    # Required because your model was trained with this column
    row["Unnamed: 0"] = frame_number

    for name, index in LANDMARKS.items():

        landmark = pose_landmarks[index]

        row[f"{name}_x"] = landmark.x
        row[f"{name}_y"] = landmark.y

    # Make sure columns are in EXACT training order
    feature_df = pd.DataFrame([row])

    feature_df = feature_df[FEATURE_COLUMNS]

    return feature_df


# ============================================================
# 6. DRAW POSE LANDMARKS
# ============================================================

connections = [
    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (28, 30),

    (29, 31),
    (30, 32)
]


def draw_landmarks(frame, landmarks):

    h, w, _ = frame.shape

    # Draw points
    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        if 0 <= x < w and 0 <= y < h:

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # Draw skeleton
    for start_idx, end_idx in connections:

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)

        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


# ============================================================
# 7. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    pose_landmarker.close()

    exit()


# Set webcam resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


frame_number = 0

previous_time = time.time()


# ============================================================
# 8. WEBCAM LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read webcam frame.")

        break


    frame_number += 1


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe Image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        time.time() * 1000
    )


    # --------------------------------------------------------
    # MediaPipe Pose Detection
    # --------------------------------------------------------

    result = pose_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # Check if person detected
    # --------------------------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


        # ----------------------------------------------------
        # Draw skeleton
        # ----------------------------------------------------

        draw_landmarks(
            frame,
            landmarks
        )


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        input_data = extract_features(
            landmarks,
            frame_number
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = np.max(
            probabilities
        ) * 100


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        text = f"Activity: {prediction}"

        confidence_text = f"Confidence: {confidence:.2f}%"


        cv2.rectangle(
            frame,
            (20, 20),
            (500, 120),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            text,
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            confidence_text,
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "No person detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "MediaPipe Human Activity Recognition",
        frame
    )


    # Press Q to exit

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 9. RELEASE EVERYTHING
# ============================================================

cap.release()

cv2.destroyAllWindows()

pose_landmarker.close()

print("Webcam stopped.")