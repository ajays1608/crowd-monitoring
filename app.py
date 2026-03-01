import os, time, cv2, sqlite3
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
from functools import wraps
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from detect import detect_persons
from density import calculate_density, generate_heatmap, assess_risk

app = Flask(__name__)
app.secret_key = 'super_secret_crowd_key'

app.jinja_env.globals.update(enumerate=enumerate)

# ✅ Absolute Paths for Windows & OpenCV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT = os.path.join(BASE_DIR, 'static', 'output')

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD
app.config['OUTPUT_FOLDER'] = OUTPUT

ALLOWED = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
def allowed(f): return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('crowd_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS crowd_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            count     INTEGER,
            limit_val INTEGER,
            status    TEXT,
            source    TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────
model_live           = None
DANGER_ZONE          = np.array([[0, 0], [1024, 0], [1024, 800], [0, 800]], np.int32)
latest_dl_count      = 0
latest_dynamic_limit = 0
limit_history        = []
calibration_frames   = 0
locked_limit         = 0

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '123'

# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USERNAME and
                request.form.get('password') == ADMIN_PASSWORD):
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# ─────────────────────────────────────────
# LIVE / VIDEO STREAM
# ─────────────────────────────────────────
def generate(source):
    global model_live, latest_dl_count, latest_dynamic_limit
    global limit_history, calibration_frames, locked_limit

    if model_live is None:
        # ✅ FIX 1: Switched to yolov8n.pt (Nano) for a massive FPS speed boost
        model_live = YOLO('yolov8n.pt')

    is_live = (str(source) == '0' or str(source).startswith('http'))

    if str(source) == '0' or source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 30) # Suggest frame rate to webcam
    else:
        cap = cv2.VideoCapture(source)
        if str(source).startswith('http'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    last_boxes  = []

    try:
        while cap.isOpened():
            if is_live:
                cap.grab() # Grab to keep buffer clean, but avoid double-grabbing overhead

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            if w > 1024:
                frame = cv2.resize(frame, (1024, int(1024 * (h / w))))
                h, w  = frame.shape[:2]

            frame_count += 1
            skip_rate = 2 if is_live else 3

            if frame_count % skip_rate == 0:
                small  = cv2.resize(frame, (640, int(640 * (h / w))))
                sh, sw = small.shape[:2]
                infer_size = 640 if is_live else 480

                # ✅ FIX 2: Increased confidence to 0.30 and iou to 0.45 to skip processing background noise
                results = model_live.predict(
                    small, classes=[0], conf=0.30, iou=0.45,
                    imgsz=infer_size, max_det=300, verbose=False
                )

                last_boxes = []
                boxes = results[0].boxes
                if boxes is not None:
                    for b in boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                        x1 = int(x1 * w / sw); y1 = int(y1 * h / sh)
                        x2 = int(x2 * w / sw); y2 = int(y2 * h / sh)
                        last_boxes.append((x1, y1, x2, y2))

            zone_count = 0
            areas      = []

            cv2.polylines(frame, [DANGER_ZONE], isClosed=True, color=(255, 0, 0), thickness=3)

            for (x1, y1, x2, y2) in last_boxes:
                cx, cy = (x1 + x2) // 2, y2
                if cv2.pointPolygonTest(DANGER_ZONE, (cx, cy), False) >= 0:
                    zone_count += 1
                    areas.append((x2 - x1) * (y2 - y1))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ✅ FIX 3: Improved Room Calibration Accuracy
            if locked_limit > 0:
                dynamic_limit = locked_limit
            else:
                dynamic_limit = latest_dynamic_limit
                if zone_count > 0 and len(areas) > 0:
                    # Filter out outlier areas (people blocking camera, or tiny false positives)
                    max_person_size = (w * h) * 0.40  # Person shouldn't take up more than 40% of screen
                    min_person_size = (w * h) * 0.005 # Ignore tiny specks
                    
                    valid_areas = [a for a in areas if min_person_size < a < max_person_size]
                    
                    if len(valid_areas) > 0:
                        median_person_area = np.median(valid_areas)
                        # Calculate exact pixel area of the polygon instead of a guess
                        zone_pixel_area = cv2.contourArea(DANGER_ZONE)
                        
                        # 80% usable space / 1.5x spacing factor instead of 3.0 padding
                        raw_limit = int((zone_pixel_area * 0.80) / (median_person_area * 1.5))
                        raw_limit = max(5, min(raw_limit, 200)) # Clamp between 5 and 200 to prevent absurd numbers
                        
                        limit_history.append(raw_limit)
                        calibration_frames += 1
                        
                        # Use median of history for stable calibration, immune to sudden spikes
                        dynamic_limit = max(1, int(np.median(limit_history)))
                        
                        if calibration_frames >= 50:
                            locked_limit = dynamic_limit
                else:
                    dynamic_limit = latest_dynamic_limit if latest_dynamic_limit > 0 else 0

            if dynamic_limit > 0:
                latest_dynamic_limit = dynamic_limit
            latest_dl_count = zone_count

            # ✅ Save to database every 1 second (30 frames)
            if frame_count % 30 == 0:
                db_status = "BREACHED" if zone_count > dynamic_limit else "WARNING" if zone_count >= int(dynamic_limit * 0.8) else "SAFE"
                src_label = "Webcam" if str(source) == '0' else "Mobile CCTV" if str(source).startswith('http') else "Video File"
                try:
                    conn = sqlite3.connect('crowd_data.db')
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO crowd_log (timestamp, count, limit_val, status, source) VALUES (?,?,?,?,?)",
                        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), zone_count, dynamic_limit, db_status, src_label)
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"DB Error: {e}")

            cv2.rectangle(frame, (10, 10), (450, 95), (0, 0, 0), -1)

            if locked_limit == 0:
                progress_pct = min(int((calibration_frames / 50.0) * 100), 100)
                cv2.putText(frame, f"CALIBRATING ROOM: {progress_pct}%",
                            (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Scanning people sizes...",
                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"IN ROOM: {zone_count} / {locked_limit}",
                            (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if zone_count > locked_limit:
                    cv2.putText(frame, "LIMIT BREACHED!", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                elif zone_count >= int(locked_limit * 0.8):
                    cv2.putText(frame, "WARNING: NEAR CAPACITY", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
                else:
                    cv2.putText(frame, "SAFE", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            quality = 65 if is_live else 75
            success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if success:
                try:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                except (GeneratorExit, Exception):
                    break
    finally:
        cap.release()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route('/video_feed')
@login_required
def video_feed():
    source = session.get('video_source', 0)
    if str(source) == '0': source = 0
    return Response(generate(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
@login_required
def live_data():
    return jsonify({'count': latest_dl_count, 'limit': latest_dynamic_limit})

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect('crowd_data.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, count, limit_val, status, source FROM crowd_log ORDER BY id DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    conn = sqlite3.connect('crowd_data.db')
    c = conn.cursor()
    c.execute("DELETE FROM crowd_log")
    conn.commit()
    conn.close()
    return redirect(url_for('history'))

# ─────────────────────────────────────────
# STATIC IMAGE LOGIC
# ─────────────────────────────────────────
def process_image(disk_filepath):
    annotated, count, coords, dynamic_limit = detect_persons(disk_filepath)

    timestamp     = int(time.time())
    ann_filename  = f"ann_{timestamp}.jpg"
    heat_filename = f"heat_{timestamp}.jpg"

    ann_disk_path  = os.path.join(OUTPUT, ann_filename)
    heat_disk_path = os.path.join(OUTPUT, heat_filename)

    cv2.imwrite(ann_disk_path, annotated)

    density = calculate_density(disk_filepath, coords)
    generate_heatmap(disk_filepath, density, heat_disk_path)

    risk, recs = assess_risk(count, density)

    # ✅ Save to database
    try:
        conn = sqlite3.connect('crowd_data.db')
        c = conn.cursor()
        db_status = "BREACHED" if "Dangerous" in risk else "WARNING" if "Crowded" in risk else "SAFE"
        c.execute(
            "INSERT INTO crowd_log (timestamp, count, limit_val, status, source) VALUES (?,?,?,?,?)",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), count, dynamic_limit, db_status, "Image Upload")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error on Image: {e}")

    return (url_for('static', filename=f'output/{ann_filename}'),
            url_for('static', filename=f'output/{heat_filename}'),
            count, risk, recs, dynamic_limit)

def reset_calibration():
    global latest_dl_count, limit_history, calibration_frames, locked_limit
    latest_dl_count = 0
    limit_history   = []
    calibration_frames = 0
    locked_limit    = 0

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    reset_calibration()
    if request.method == 'POST':

        if request.form.get('ip_url'):
            session['video_source'] = request.form.get('ip_url')
            return render_template('index.html', mode='video')

        file = request.files.get('file')
        if file and allowed(file.filename):

            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'jpg'
            safe_name     = f"upload_{int(time.time())}.{ext}"
            disk_filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], safe_name))

            file.save(disk_filepath)

            if not os.path.exists(disk_filepath) or os.path.getsize(disk_filepath) == 0:
                return f"<h2>ERROR: File failed to save. Check that static/uploads/ folder exists in your project!</h2>"

            if ext in {'mp4', 'avi', 'mov'}:
                session['video_source'] = disk_filepath
                return render_template('index.html', mode='video')
            else:
                ann_url, heat_url, count, risk, recs, img_limit = process_image(disk_filepath)
                orig_url = url_for('static', filename=f'uploads/{safe_name}')

                return render_template('index.html', mode='image',
                                       original=orig_url,
                                       annotated=ann_url,
                                       heatmap=heat_url,
                                       count=count,
                                       risk=risk,
                                       recs=recs,
                                       limit=img_limit)

    return render_template('index.html', mode='upload')

@app.route('/webcam')
@login_required
def webcam():
    reset_calibration()
    session['video_source'] = 0
    return render_template('index.html', mode='video')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
