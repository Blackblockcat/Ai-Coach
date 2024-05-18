import cv2
import mediapipe as mp
import numpy as np
from keras import models
from keras.utils import plot_model
import time

model = models.load_model(
    r"C:\Users\Deiaa\OneDrive\Desktop\Games\AI Coach\model.h5")

cap = cv2.VideoCapture(
    "C:\Users\Deiaa\OneDrive\Desktop\Games\AI Coach\\test4.mp4")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
# correct_predictions = 0
# total_predictions = 0
timer_start = None
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            results.pose_landmarks,  # model output
            mp_pose.POSE_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_pose_landmarks_style())

        for i in range(len(results.pose_landmarks.landmark)):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(results.pose_landmarks.landmark)):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        # print(np.array(data_aux))
        data_aux = np.array(data_aux).reshape(1, 66)
        prediction = model.predict(data_aux)
        predicted_label = round(float(prediction))
        print(predicted_label)

        if predicted_label == 1:
            text = 'false'
            if timer_start is not None:
                timer_start = None
        else:
            text = 'true'
            if timer_start is None:
                timer_start = time.time()

        if timer_start is not None:
            elapsed_time = time.time() - timer_start
            cv2.putText(frame, f'Time: {int(elapsed_time)}s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        if text=='true':
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        else:
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
