import cv2
import pandas as pd
from datetime import datetime

# Ask for the user's name before starting
name = input("Enter your name: ")

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create or load attendance CSV
attendance_file = "attendance.csv"
try:
    attendance_df = pd.read_csv(attendance_file)
except FileNotFoundError:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

# Function to mark attendance
def mark_attendance(person_name):
    global attendance_df
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_df.loc[len(attendance_df)] = [person_name, current_time]
    attendance_df.to_csv(attendance_file, index=False)
    print(f"✅ Attendance marked for {person_name} at {current_time}")

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting camera... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Detection Attendance', frame)

    # When face detected, mark attendance and break
    if len(faces) > 0:
        mark_attendance(name)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Attendance saved to attendance.csv")
