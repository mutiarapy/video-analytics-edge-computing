import cv2
import numpy as np
import pickle
import threading
import queue
import os
import subprocess
import requests
import time
import torch
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response, render_template, send_from_directory
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('py.env')

USE_GPU = torch.cuda.is_available()
print(f"[INFO] GPU: {'ON — ' + torch.cuda.get_device_name(0) if USE_GPU else 'OFF'}")

import onnxruntime as _ort
from insightface.app import FaceAnalysis

_available_providers = _ort.get_available_providers()
print(f"[INFO] Available ORT providers: {_available_providers}")

if 'TensorrtExecutionProvider' in _available_providers:
    print("[INFO] InsightFace: using TensorrtExecutionProvider")
    providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
    ctx_id = 0
elif 'CUDAExecutionProvider' in _available_providers:
    print("[INFO] InsightFace: using CUDAExecutionProvider")
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    ctx_id = 0
else:
    print("[INFO] InsightFace: CUDAExecutionProvider unavailable, using CPU")
    print("[INFO] (Known Jetson Orin Nano onnxruntime limitation — YOLO still uses GPU)")
    providers = ['CPUExecutionProvider']
    ctx_id = -1

YOLO_ENGINE_PATHS = [
    '/app/models/yolo11n.engine',
    'yolo11n.engine',
]
YOLO_PT = 'yolo11n.pt'

engine_found = None
for path in YOLO_ENGINE_PATHS:
    if os.path.exists(path):
        engine_found = path
        break

if engine_found:
    print(f"[INFO] YOLO: Loading TensorRT engine → {engine_found}")
    model = YOLO(engine_found)
else:
    print(f"[WARN] TensorRT engine tidak ditemukan, fallback → {YOLO_PT}")
    model = YOLO(YOLO_PT)
    if USE_GPU:
        model.to('cuda')

with open("dataset_wajah.pkl", "rb") as f:
    data             = pickle.load(f)
    known_embeddings = np.array(data["embeddings"])
    known_names      = data["names"]

if USE_GPU:
    db_tensor = torch.tensor(known_embeddings, device='cuda').float()

face_app = FaceAnalysis(name='buffalo_s', providers=providers)
face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

print(f"[INFO] InsightFace dataset: {len(known_names)} wajah")

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

app = Flask(__name__)

PYTHON_SERVER_URL = os.getenv("PYTHON_SERVER_URL", "http://localhost:5000")
SERVER_URL        = os.getenv("SERVER_URL",        "http://localhost:3000")
STREAM_URL        = os.getenv("STREAM_URL")
if not STREAM_URL:
    raise ValueError("STREAM_URL not set in py.env")

save_dir  = os.getenv("SAVE_DIR",  "/recordings")
clips_dir = os.getenv("CLIPS_DIR", "/clips")
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(clips_dir, exist_ok=True)

BUFFER_SIZE       = 150
CLIP_DURATION_SEC = 10
#CLIP_COOLDOWN     = 15
REPORT_COOLDOWN   = 15
MIN_VERTICAL_MOVE = 10
YOLO_EVERY        = 5
FACE_EVERY        = 25

class BoundaryLine:
    def __init__(self, p0=(0, 0), p1=(0, 0)):
        self.p0, self.p1 = p0, p1
        self.count1 = self.count2 = 0

fenceLine      = BoundaryLine(p0=(230, 120), p1=(360, 120))
prev_positions = {}
crossed_ids    = set()

latest_frame     = None
frame_lock       = threading.Lock()

latest_cam_frame = None
cam_frame_lock   = threading.Lock()
cam_ready        = threading.Event()

face_overlay      = []
face_overlay_lock = threading.Lock()

face_queue = queue.Queue(maxsize=1)

frame_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock  = threading.Lock()

_report_lock   = threading.Lock()
_clip_lock     = threading.Lock()
last_reported  = {}
last_clip_time = {}

def should_report(nama):
    with _report_lock:
        now = time.time()
        if nama not in last_reported or (now - last_reported[nama]) > REPORT_COOLDOWN:
            last_reported[nama] = now
            return True
        return False

def open_camera():
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap

def resize_frame(frame, width=854, height=480):
    if USE_GPU:
        try:
            g = cv2.cuda_GpuMat()
            g.upload(frame)
            g = cv2.cuda.resize(g, (width, height))
            return g.download()
        except cv2.error:
            pass
    return cv2.resize(frame, (width, height))

def get_writer():
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    filepath = os.path.join(save_dir, filename)
    writer   = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (854, 480))
    print(f"[REC] Started: {filepath}")
    return writer, filepath

def convert_video(filepath):
    out    = filepath.replace('.mp4', '_fixed.mp4')
    result = subprocess.run(
        ['ffmpeg', '-i', filepath, '-c:v', 'libx264',
         '-preset', 'fast', '-movflags', '+faststart', '-y', out],
        capture_output=True
    )
    if result.returncode == 0:
        os.remove(filepath)
        os.rename(out, filepath)
        print(f"[REC] Converted: {filepath}")
    else:
        print(f"[REC] FFmpeg error: {result.stderr.decode()}")

def notify_recording(filepath):
    try:
        filename = os.path.basename(filepath)
        with open(filepath, 'rb') as f:
            requests.post(
                f"{SERVER_URL}/upload-video",
                files={'video': (filename, f, 'video/mp4')},
                data={'event_type': 'Recording', 'duration_sec': 1800},
                timeout=30
            )
        print(f"[REC] Sent: {filename}")
    except Exception as e:
        print(f"[REC] notify error: {e}")

def report_detection(nama, confidence, image_path=" ", direction=None):
    if "Crossing" in nama:
        event_type = "Anomali"
    elif nama == "Unknown":
        event_type = "Anomali"
    else:
        event_type = "Terdeteksi"

    payload = {"nama": nama, "confidence": float(confidence),
               "image_path": image_path, "event_type": event_type}
    if direction is not None:
        payload["direction"] = direction

    for attempt in range(3):
        try:
            resp   = requests.post(f"{SERVER_URL}/report-anomaly", json=payload, timeout=5)
            result = resp.json()
            det_id = result.get("id")
            print(f"[RPT] {event_type} — {nama} (id={det_id})")
            return det_id, event_type
        except Exception as e:
            print(f"[RPT] attempt {attempt+1} error: {e}")
            time.sleep(0.5)

    print(f"[RPT] Gagal setelah 3 percobaan: {nama}")
    return None, event_type

def save_clip(trigger_reason, frames_snapshot, detection_id=None):
    try:
        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"CLIP_{trigger_reason}_{ts}.mp4"
        filepath = os.path.join(clips_dir, filename)
        h, w     = frames_snapshot[0].shape[:2]
        out      = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
        for fr in frames_snapshot:
            out.write(fr)
        out.release()
        convert_video(filepath)
        requests.post(f"{SERVER_URL}/notify-clip", json={
            "filename": filename, "trigger": trigger_reason,
            "clip_url": f"{PYTHON_SERVER_URL}/clips/{filename}",
            "timestamp": datetime.now().isoformat(),
            "duration_sec": CLIP_DURATION_SEC, "detection_id": detection_id
        }, timeout=5)
        print(f"[CLIP] Sent: {filename}")
    except Exception as e:
        print(f"[CLIP] error: {e}")

def trigger_clip(reason, det_id=None):
    with buffer_lock:
        frames_copy = list(frame_buffer)[-(CLIP_DURATION_SEC * 15):]
    if len(frames_copy) < 30:
        print(f"[CLIP] Buffer kurang ({len(frames_copy)} frames), skip")
        return
    threading.Thread(target=save_clip, args=(reason, frames_copy, det_id), daemon=True).start()

def camera_reader():
    global latest_cam_frame
    cam        = open_camera()
    fail_count = 0

    while True:
        success, frame = cam.read()
        if not success:
            fail_count += 1
            if fail_count >= 5:
                print("[CAM] Reconnecting...")
                cam.release()
                time.sleep(2)
                cam        = open_camera()
                fail_count = 0
            continue
        fail_count = 0
        with cam_frame_lock:
            latest_cam_frame = frame
        cam_ready.set()

def face_worker():
    print("[INFO] Face worker started (InsightFace + buffalo_s)")
    while True:
        try:
            frame = face_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            faces      = face_app.get(frame)
            detections = []

            for face in faces:
                if face.det_score < 0.65:
                    continue

                bbox  = face.bbox.astype(int)
                embed = face.normed_embedding

                if USE_GPU:
                    embed_t = torch.tensor(embed, device='cuda').float()
                    sims    = torch.mv(db_tensor, embed_t).cpu().numpy().tolist()
                else:
                    sims = [np.dot(embed, e) for e in known_embeddings]

                if sims and max(sims) > 0.4:
                    idx   = int(np.argmax(sims))
                    score = max(sims)
                    name  = known_names[idx]
                    label = f"{name} ({score:.2f})"
                    color = (0, 255, 0)
                    # FIX: should_report dipanggil di sini, bukan di dalam report_detection
                    if should_report(name):
                        threading.Thread(
                            target=report_detection,
                            args=(name, score),
                            daemon=True
                        ).start()
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    # FIX: indentasi benar, should_report dipanggil di sini
                    if should_report("Unknown"):
                        def _unk():
                            did, _ = report_detection("Unknown", 0.0)
                            if did:
                                trigger_clip("UNKNOWN", did)
                        threading.Thread(target=_unk, daemon=True).start()

                detections.append((bbox[0], bbox[1], bbox[2], bbox[3], label, color))

            with face_overlay_lock:
                face_overlay.clear()
                face_overlay.extend(detections)

        except Exception as e:
            print(f"[face_worker] error: {e}")

def process_frames():
    global latest_frame

    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.6)

    counter          = 0
    results          = []
    writer, filepath = get_writer()
    frames_written   = 0
    MAX_FRAMES       = 15 * 60 * 30

    while True:
        try:
            if not cam_ready.wait(timeout=5.0):
                print("[MAIN] Timeout nunggu frame kamera...")
                continue

            with cam_frame_lock:
                if latest_cam_frame is None:
                    continue
                frame = latest_cam_frame.copy()
   

            counter += 1
            frame    = resize_frame(frame, 854, 480)

            with buffer_lock:
                frame_buffer.append(frame.copy())

            if counter % YOLO_EVERY == 0:
                results = model(frame, verbose=False, classes=[0])

            if counter % FACE_EVERY == 0:
                try:
                    face_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            with face_overlay_lock:
                overlay_snapshot = list(face_overlay)
            for (x1, y1, x2, y2, label, color) in overlay_snapshot:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                fence_y = fenceLine.p0[1]
                x_min   = min(fenceLine.p0[0], fenceLine.p1[0])
                x_max   = max(fenceLine.p0[0], fenceLine.p1[0])

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        matched_id = None
                        min_dist   = 80
                        for tid, (pcx, pcy, py1, py2) in prev_positions.items():
                            dist = abs(pcx - cx) + abs(pcy - cy)
                            if dist < min_dist:
                                matched_id = tid
                                min_dist   = dist
                        if matched_id is None:
                            matched_id = max(prev_positions.keys(), default=-1) + 1

                        if matched_id in prev_positions:
                            pcx, pcy, py1, py2 = prev_positions[matched_id]
                            in_zone      = x_min <= cx <= x_max
                            bbox_crosses = y1 < fence_y < y2
                            vert_move    = abs(cy - pcy)

                            if in_zone and bbox_crosses and matched_id not in crossed_ids:
                                if pcy > fence_y and vert_move >= MIN_VERTICAL_MOVE:
                                    fenceLine.count1 += 1
                                    crossed_ids.add(matched_id)
                                    print(f"[LINE] MASUK! Total: {fenceLine.count1}")
                                    def _masuk(d="masuk"):
                                        did, _ = report_detection(f"Crossing-{d}", 1.0, direction=d)
                                        trigger_clip(d.upper(), did)
                                    threading.Thread(target=_masuk, daemon=True).start()

                                elif pcy < fence_y and vert_move >= MIN_VERTICAL_MOVE:
                                    fenceLine.count2 += 1
                                    crossed_ids.add(matched_id)
                                    print(f"[LINE] KELUAR! Total: {fenceLine.count2}")
                                    def _keluar(d="keluar"):
                                        did, _ = report_detection(f"Crossing-{d}", 1.0, direction=d)
                                        trigger_clip(d.upper(), did)
                                    threading.Thread(target=_keluar, daemon=True).start()

                        prev_positions[matched_id] = (cx, cy, y1, y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                if counter % 75 == 0:
                    active_ids = set()
                    if results:
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx, cy = (x1+x2)//2, (y1+y2)//2
                                for tid, (pcx, pcy, *_) in prev_positions.items():
                                    if abs(pcx - cx) + abs(pcy - cy) < 80:
                                        active_ids.add(tid)
                    for sid in set(prev_positions.keys()) - active_ids:
                        del prev_positions[sid]
                        crossed_ids.discard(sid)

                #cv2.line(frame, fenceLine.p0, fenceLine.p1, (0, 255, 255), 3)

                writer.write(frame)
                frames_written += 1
                if frames_written >= MAX_FRAMES:
                    writer.release()
                    def _finish(fp):
                        convert_video(fp)
                        notify_recording(fp)
                    threading.Thread(target=_finish, args=(filepath,), daemon=True).start()
                    writer, filepath = get_writer()
                    frames_written   = 0

            _, buffer = cv2.imencode('.jpg', frame)
            with frame_lock:
                latest_frame = buffer.tobytes()

        except Exception as e:
            print(f"[MAIN] error: {e}")
            continue

threading.Thread(target=camera_reader,  daemon=True).start()
threading.Thread(target=face_worker,    daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clips/<filename>')
def serve_clip(filename):
    return send_from_directory(clips_dir, filename)

@app.route('/list-clips')
def list_clips():
    return {'clips': sorted(os.listdir(clips_dir), reverse=True)}

@app.route('/recordings')
def list_recordings():
    return {'recordings': sorted(os.listdir(save_dir), reverse=True)}

@app.route('/recordings/<filename>')
def serve_recording(filename):
    return send_from_directory(save_dir, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)