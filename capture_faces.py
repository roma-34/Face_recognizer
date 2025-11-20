import cv2, os

name = input("Enter your name: ")
folder = f"dataset/{name}"
os.makedirs(folder, exist_ok=True)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0
print("📸 Capturing 50 images. Look at the camera...")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{folder}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Capturing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

print(f"✅ Captured {count} images for {name}")
cam.release()
cv2.destroyAllWindows()
