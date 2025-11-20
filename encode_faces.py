import cv2
import face_recognition
import os
import pickle

# Path to folder containing images
path = 'images'
images = []
names = []

# Load images
for file in os.listdir(path):
    if file.endswith(('jpg', 'jpeg', 'png')):
        img = cv2.imread(f'{path}/{file}')
        images.append(img)
        names.append(os.path.splitext(file)[0])

def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        encode_list.append(enc)
    return encode_list

print("[INFO] Encoding faces...")
encodeListKnown = findEncodings(images)
print(f"[INFO] Encoded {len(encodeListKnown)} faces")

# Save encodings
with open('encodings.pkl', 'wb') as f:
    pickle.dump((encodeListKnown, names), f)

print("[SUCCESS] Encodings saved to encodings.pkl")
