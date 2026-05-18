from ultralytics import YOLO
import os
import shutil

print("[TRT] Mulai export YOLO ke TensorRT...")
print("[TRT] Proses ini butuh 3-10 menit di Jetson, harap tunggu...")

MODEL_DIR  = '/app/models'
ENGINE_OUT = os.path.join(MODEL_DIR, 'yolo11n.engine')
os.makedirs(MODEL_DIR, exist_ok=True)

# Kalau engine sudah ada, skip
if os.path.exists(ENGINE_OUT):
    size_mb = os.path.getsize(ENGINE_OUT) / 1024 / 1024
    print(f"[TRT] Engine sudah ada ({size_mb:.1f} MB), skip export")
else:
    model = YOLO('yolo11n.pt')
    model.export(
        format='engine',
        half=True,
        device=0,
        workspace=2,
        verbose=True
    )
    # Ultralytics menyimpan engine di direktori kerja
    if os.path.exists('yolo11n.engine'):
        shutil.move('yolo11n.engine', ENGINE_OUT)
        print(f"[TRT] Sukses! Engine disimpan ke {ENGINE_OUT}")
    else:
        print("[TRT] GAGAL — engine tidak terbuat")