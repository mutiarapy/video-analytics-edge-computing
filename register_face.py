import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# 1. Inisialisasi InsightFace
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

# 2. Tentukan lokasi folder dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset")
known_embeddings = []
known_names = []

print("Sedang memproses foto wajah... mohon tunggu.")

# 3. Loop setiap folder (nama orang) di dalam dataset
for name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, name)
    
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        # Ekstrak embedding wajah
        faces = face_app.get(img)
        
        if len(faces) > 0:
            # Ambil embedding dari wajah pertama yang terdeteksi di foto
            emb = faces[0].normed_embedding
            known_embeddings.append(emb)
            known_names.append(name)
            print(f"Berhasil memproses: {name} ({image_name})")
        else:
            print(f"Wajah tidak ditemukan di foto: {image_path}")

# 4. Simpan ke file .pkl
data = {"embeddings": known_embeddings, "names": known_names}
with open("dataset_wajah.pkl", "wb") as f:
    pickle.dump(data, f)


print("\nSelesai! File 'dataset_wajah.pkl' telah dibuat.")