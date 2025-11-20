import cv2, os, numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces, ids, names = [], [], []
current_id = 0
name_map = {}

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder): continue
    name_map[current_id] = person
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            faces.append(gray)
            ids.append(current_id)
    current_id += 1

recognizer.train(faces, np.array(ids))
recognizer.write("trainer.yml")

with open("names.txt", "w") as f:
    for k, v in name_map.items():
        f.write(f"{k},{v}\n")

print("✅ Training complete. Model saved to trainer.yml")
