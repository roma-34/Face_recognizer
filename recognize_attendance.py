import cv2, pandas as pd
from datetime import datetime

# Load model and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
name_map = {int(k): v for k,v in [line.strip().split(",") for line in open("names.txt")]}

attendance_file = "attendance.csv"
try:
    df = pd.read_csv(attendance_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name","Time"])

def mark(name):
    global df
    if name not in df["Name"].values:
        now = datetime.now().strftime("%H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(attendance_file, index=False)
        print(f"✅ {name} marked at {now}")

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("🎥 Recognizing... Press 'q' to quit")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            name = name_map.get(id_, "Unknown")
            color = (0,255,0)
            mark(name)
        else:
            name = "Unknown"
            color = (0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("👋 Attendance saved to attendance.csv")
