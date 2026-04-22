import cv2
import numpy as np
import pickle
import threading
import os
import subprocess
import requests
import time
from collections import deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from flask import Flask, Response, render_template, send_from_directory
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('py.env')

#garis
class BoundaryLine:
    def __init__(self, p0=(0, 0), p1=(0, 0)):
        self.p0 = p0
        self.p1 = p1
        self.count1 = 0
        self.count2 = 0  

fenceLine      = BoundaryLine(p0=(230, 120), p1=(360, 120))
prev_positions = {}   
crossed_ids    = set()

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

app = Flask(__name__)

PYTHON_SERVER_URL = os.getenv("PYTHON_SERVER_URL", "http://localhost:5000")
SERVER_URL        = os.getenv("SERVER_URL", "http://localhost:3000")

model = YOLO('yolov8n.pt')

with open("dataset_wajah.pkl", "rb") as f:
    data             = pickle.load(f)
    known_embeddings = data["embeddings"]
    known_names      = data["names"]

face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

STREAM_URL = os.getenv("STREAM_URL")
if not STREAM_URL:
    raise ValueError("STREAM_URL not set..")

def open_camera():
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap

latest_frame = None
frame_lock   = threading.Lock()

save_dir          = os.getenv("SAVE_DIR",           "/home/isrdds/video_analytics/recordings")
clips_dir         = os.getenv("CLIPS_DIR",          "/home/isrdds/video_analytics/clips")
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(clips_dir, exist_ok=True)

BUFFER_SIZE       = 150   
CLIP_DURATION_SEC = 10
CLIP_COOLDOWN     = 15
REPORT_COOLDOWN   = 10
MIN_VERTICAL_MOVE = 10    # pixel minimum gerak vertikal agar dihitung crossing

last_reported  = {}
last_clip_time = 0

frame_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock  = threading.Lock()

def should_report(nama):
    now = time.time()
    if nama not in last_reported or (now - last_reported[nama]) > REPORT_COOLDOWN:
        last_reported[nama] = now
        return True
    return False

def get_writer():
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    filepath = os.path.join(save_dir, filename)
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    writer   = cv2.VideoWriter(filepath, fourcc, 15.0, (854, 480))
    print(f"Recording started: {filepath}")
    return writer, filepath

def convert_video(filepath):
    output = filepath.replace('.mp4', '_fixed.mp4')
    result = subprocess.run(
        ['ffmpeg', '-i', filepath,
         '-c:v', 'libx264',
         '-preset', 'fast',
         '-movflags', '+faststart',
         '-y', output],
        capture_output=True
    )
    if result.returncode == 0:
        os.remove(filepath)
        os.rename(output, filepath)
        print(f"Berhasil Konversi & Optimize: {filepath}")
    else:
        print(f"FFmpeg Error: {result.stderr.decode()}")

def report_detection(nama, confidence, image_path="foto.jpg", direction=None):
    try:
        if "Crossing" in nama:
            event_type = "Line Crossing"
        elif nama == "Unknown":
            event_type = "Anomali"
        else:
            event_type = "Terdeteksi"

        payload = {
            "nama":       nama,
            "confidence": float(confidence),
            "image_path": image_path,
            "event_type": event_type,
        }
        if direction is not None:
            payload["direction"] = direction  

        resp   = requests.post(f"{SERVER_URL}/report-anomaly", json=payload, timeout=5)
        result = resp.json()
        print(f"Saved: {event_type} - {nama} | direction={direction} (id={result.get('id')})")
        return result.get("id"), event_type
    except Exception as e:
        print(f"report_detection error: {e}")
        return None, "Anomali"

# buat nyimpen clip 10 detik ke /clips 
def save_clip(trigger_reason, frames_snapshot, detection_id=None):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename  = f"CLIP_{trigger_reason}_{timestamp}.mp4"
        filepath  = os.path.join(clips_dir, filename)

        h, w = frames_snapshot[0].shape[:2]
        out  = cv2.VideoWriter(
            filepath,
            cv2.VideoWriter_fourcc(*'mp4v'),
            15, (w, h)
        )
        for f in frames_snapshot:
            out.write(f)
        out.release()

        convert_video(filepath)

        clip_url = f"{PYTHON_SERVER_URL}/clips/{filename}"
        requests.post(f"{SERVER_URL}/notify-clip", json={
            "filename":     filename,
            "trigger":      trigger_reason,
            "clip_url":     clip_url,
            "timestamp":    datetime.now().isoformat(),
            "duration_sec": CLIP_DURATION_SEC,
            "detection_id": detection_id
        }, timeout=5)
        print(f"Clip sent to Node.js: {filename}")

    except Exception as e:
        print(f"save_clip error: {e}")

def trigger_clip(reason, det_id=None):
    global last_clip_time
    now = time.time()
    if (now - last_clip_time) < CLIP_COOLDOWN:
        return
    last_clip_time = now

    with buffer_lock:
        frames_copy = list(frame_buffer)[-(CLIP_DURATION_SEC * 15):]

    if len(frames_copy) < 30:
        return

    threading.Thread(
        target=save_clip,
        args=(reason, frames_copy, det_id),
        daemon=True
    ).start()

def process_frames():
    global latest_frame

    cam = open_camera()
    fail_count = 0

    counter          = 0
    results          = []
    faces            = []
    writer, filepath = get_writer()
    frames_written   = 0
    MAX_FRAMES       = 15 * 60 * 30  

    while True:
        try:
            success, frame = cam.read()

            # auto reconnect kamera
            if not success:
                fail_count += 1
                print(f"Gagal mengambil frame ({fail_count}x)...")
                if fail_count >= 5:
                    print("Reconnecting kamera...")
                    cam.release()
                    time.sleep(2)
                    cam = open_camera()
                    fail_count = 0
                continue

            fail_count = 0
            counter   += 1
            frame      = cv2.resize(frame, (854, 480))

            with buffer_lock:
                frame_buffer.append(frame.copy())

            if counter % 5 == 0:
                results = model(frame, verbose=False, classes=[0])
                try:
                    faces = face_app.get(frame)
                except Exception:
                    faces = []

            #Proses deteksi wajah 
            for face in faces:
                if face.det_score < 0.65:
                    continue

                name  = "Unknown"
                color = (0, 0, 255)
                bbox  = face.bbox.astype(int)
                embed = face.normed_embedding
                sims  = [np.dot(embed, e) for e in known_embeddings]

                if sims and max(sims) > 0.2:
                    idx   = np.argmax(sims)
                    score = max(sims)
                    name  = f"{known_names[idx]} ({score:.2f})"
                    color = (0, 255, 0)
                    if should_report(name):
                        threading.Thread(
                            target=report_detection,
                            args=(name, score),
                            daemon=True
                        ).start()
                else:
                    if should_report("Unknown"):
                        det_id, _ = report_detection("Unknown", 0.0)
                        if det_id:
                            threading.Thread(
                                target=trigger_clip,
                                args=("UNKNOWN", det_id),
                                daemon=True
                            ).start()

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # proses bounding box orang + crossing garis
            fence_y = fenceLine.p0[1]
            x_min   = min(fenceLine.p0[0], fenceLine.p1[0])
            x_max   = max(fenceLine.p0[0], fenceLine.p1[0])

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx        = (x1 + x2) // 2
                    cy_center = (y1 + y2) // 2

                    matched_id = None
                    min_dist   = 80
                    for tid, (prev_cx, prev_cy, prev_y1, prev_y2) in prev_positions.items():
                        dist = abs(prev_cx - cx) + abs(prev_cy - cy_center)
                        if dist < min_dist:
                            matched_id = tid
                            min_dist   = dist
                    if matched_id is None:
                        matched_id = max(prev_positions.keys(), default=-1) + 1

                    if matched_id in prev_positions:
                        prev_cx, prev_cy, prev_y1, prev_y2 = prev_positions[matched_id]

                        in_zone       = x_min <= cx <= x_max
                        bbox_crosses  = y1 < fence_y < y2
                        was_above     = prev_cy < fence_y
                        was_below     = prev_cy > fence_y
                        vertical_move = abs(cy_center - prev_cy)

                        if in_zone and bbox_crosses and matched_id not in crossed_ids:

                            # MASUK: orang naik dari bawah frame melewati garis
                            if was_below and vertical_move >= MIN_VERTICAL_MOVE:
                                fenceLine.count1 += 1
                                crossed_ids.add(matched_id)
                                print(f"MASUK! Total: {fenceLine.count1}")

                                def handle_masuk(direction="masuk"):
                                    det_id, _ = report_detection(
                                        f"Crossing-{direction}",
                                        1.0,
                                        direction=direction  
                                    )
                                    trigger_clip(direction.upper(), det_id)
                                threading.Thread(target=handle_masuk, daemon=True).start()

                            # KELUAR: orang turun dari atas frame melewati garis
                            elif was_above and vertical_move >= MIN_VERTICAL_MOVE:
                                fenceLine.count2 += 1
                                crossed_ids.add(matched_id)
                                print(f"KELUAR! Total: {fenceLine.count2}")

                                def handle_keluar(direction="keluar"):
                                    det_id, _ = report_detection(
                                        f"Crossing-{direction}",
                                        1.0,
                                        direction=direction   
                                    )
                                    trigger_clip(direction.upper(), det_id)
                                threading.Thread(target=handle_keluar, daemon=True).start()

                    prev_positions[matched_id] = (cx, cy_center, y1, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # bersihkan ID lama setiap ~5 detik 
            if counter % 75 == 0:
                active_ids = set()
                if results:
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            for tid, (pcx, pcy, py1, py2) in prev_positions.items():
                                if abs(pcx - cx) + abs(pcy - cy) < 80:
                                    active_ids.add(tid)

                stale_ids = set(prev_positions.keys()) - active_ids
                for sid in stale_ids:
                    del prev_positions[sid]
                    crossed_ids.discard(sid)

            cv2.line(frame, fenceLine.p0, fenceLine.p1, (0, 255, 255), 3)

            total = fenceLine.count1 + fenceLine.count2
            cv2.putText(frame, f'Terdeteksi: {total}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            writer.write(frame)
            frames_written += 1
            if frames_written >= MAX_FRAMES:
                writer.release()
                threading.Thread(target=convert_video, args=(filepath,), daemon=True).start()
                writer, filepath = get_writer()
                frames_written   = 0

            # update frame untuk MJPEG stream 
            ret, buffer = cv2.imencode('.jpg', frame)
            with frame_lock:
                latest_frame = buffer.tobytes()

        except Exception as e:
            print(f"Error in process_frames: {e}")
            continue

thread = threading.Thread(target=process_frames, daemon=True)
thread.start()

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
    files = sorted(os.listdir(clips_dir), reverse=True)
    return {'clips': files}

@app.route('/recordings')
def list_recordings():
    files = sorted(os.listdir(save_dir), reverse=True)
    return {'recordings': files}

@app.route('/recordings/<filename>')
def serve_recording(filename):
    return send_from_directory(save_dir, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
