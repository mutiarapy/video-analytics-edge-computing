import torch
import pickle
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device,
    post_process=False,   # Output: float tensor [0, 255]
    keep_all=False,       # Ambil 1 wajah terbesar per foto
    min_face_size=40,
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def preprocess_face_tensor(face_tensor):
    """
    Konversi output MTCNN (post_process=False) → input valid InceptionResnetV1.
    MTCNN dengan post_process=False menghasilkan tensor [0, 255].
    InceptionResnetV1 mengharapkan input ternormalisasi ke [-1, 1].

    PENTING: Fungsi ini harus identik dengan yang ada di videoanalytics.py
    supaya embedding di dataset dan di inference konsisten.
    """
    face = face_tensor.float() / 255.0   # [0, 255] → [0.0, 1.0]
    face = (face - 0.5) / 0.5            # [0.0, 1.0] → [-1.0, 1.0]
    return face


dataset_dir = "/home/isrdds/app/dataset"
embeddings  = []
names       = []

for person in sorted(os.listdir(dataset_dir)):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    person_count = 0
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_dir, img_file)
        img      = Image.open(img_path).convert('RGB')
        face     = mtcnn(img)   # Shape: [3, 160, 160], dtype float32 [0, 255]

        if face is not None:
            # FIX 1: Normalisasi input ke [-1, 1] sebelum masuk resnet
            face_input = preprocess_face_tensor(face).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet(face_input)
                # FIX 2: Normalisasi embedding ke unit vector untuk cosine similarity
                emb = F.normalize(emb, p=2, dim=1)

            embeddings.append(emb.cpu().numpy()[0])
            names.append(person)
            person_count += 1
            print(f"[OK] {person} - {img_file}")
        else:
            print(f"[SKIP] Wajah tidak terdeteksi: {person} - {img_file}")

    if person_count > 0:
        print(f"  → {person_count} foto berhasil untuk '{person}'")
    else:
        print(f"  → [WARN] Tidak ada foto valid untuk '{person}'")

print()
if embeddings:
    with open("/home/isrdds/app/video-analytics-edge-computing/dataset_wajah_facenet.pkl", "wb") as f:
        pickle.dump({"embeddings": np.array(embeddings), "names": names}, f)
    print(f"Dataset saved! Total: {len(names)} embedding dari {len(set(names))} orang")

    # Verifikasi: cek shape dan norm embedding
    emb_array = np.array(embeddings)
    norms = np.linalg.norm(emb_array, axis=1)
    print(f"Embedding shape : {emb_array.shape}")
    print(f"Embedding norms : min={norms.min():.4f}, max={norms.max():.4f} (harusnya ~1.0)")
else:
    print("Tidak ada wajah yang berhasil diproses!")