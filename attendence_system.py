import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
import os
from datetime import datetime

# Load known faces
with open('encodings.pkl', 'rb') as f:
    encodeListKnown, names = pickle.load(f)

# Create attendance directory if missing
if not os.path.exists('attendance_logs'):
    os.makedirs('attendance_logs')

today_date = datetime.now().strftime("%Y-%m-%d")
csv_path = f"attendance_logs/attendance_{today_date}.csv"

if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(csv_path, index=False)

def markAttendance(name):
    df = pd.read_csv(csv_path)
    if name not in df["Name"].values:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        df.loc[len(df)] = [name, time_str]
        df.to_csv(csv_path, index=False)
        print(f"[MARKED] {name} at {time_str}")

cap = cv2.VideoCapture(0)
motion_detector = cv2.createBackgroundSubtractorMOG2()

print("[INFO] Starting camera... Press 'Q' to quit.")
while True:
    success, frame = cap.read()
    if not success:
        break

    motion_mask = motion_detector.apply(frame)
    motion_level = np.sum(motion_mask) / 255

    # Trigger recognition only if motion detected
    if motion_level > 10000:
        imgS = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                markAttendance(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] System closed.")
