import cv2
from ultralytics import YOLO
import requests
import time
import os
import threading
import queue
import numpy as np
from deepface import DeepFace
from pillow_heif import register_heif_opener

# ─── Model Loading ────────────────────────────────────────────────────────────
print("Loading YOLO models...")
pose_model = YOLO("yolov8n-pose.pt")
obj_model  = YOLO("yolov8n.pt")
print("Models loaded successfully.")

register_heif_opener()

# ─── Auto-clean bad employee photos on startup ────────────────────────────────
def clean_employee_database(db_path: str = "employees"):
    """
    Scan every photo in the employees folder.
    Delete any image where OpenCV cannot detect a face —
    these would poison the DeepFace database and cause errors.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    removed = 0
    kept    = 0
    print("\n─── Scanning employee database for bad photos... ───────────────")
    for person in sorted(os.listdir(db_path)):
        person_dir = os.path.join(db_path, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(person_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"  ✗ Unreadable, deleting:  {person}/{fname}")
                os.remove(fpath)
                removed += 1
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if len(faces) == 0:
                print(f"  ✗ No face detected, deleting: {person}/{fname}")
                os.remove(fpath)
                removed += 1
            else:
                kept += 1
    print(f"  ✓ Kept {kept} valid photo(s), removed {removed} bad photo(s).")
    # If any photos were removed, wipe the pkl cache so DB rebuilds cleanly
    if removed > 0:
        for f in os.listdir(db_path):
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(db_path, f))
                    print(f"  ✓ Cache cleared for clean rebuild.")
                except Exception:
                    pass
    print("────────────────────────────────────────────────────────────────\n")

clean_employee_database()

cap = cv2.VideoCapture(0)

BACKEND_URL      = "http://localhost:8000/update"
UPDATE_INTERVAL  = 1.0
RESCAN_INTERVAL  = 15.0


# ─── Threaded Face Recognizer ─────────────────────────────────────────────────
class FaceRecognizer:
    def __init__(self, db_path="employees"):
        self.db_path     = db_path
        self.input_queue = queue.Queue(maxsize=5)
        self.results     = {}
        self.processing  = set()
        self.lock        = threading.Lock()
        self.thread      = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _refresh_cache_if_needed(self):
        """Delete stale .pkl if employees folder is newer."""
        pkl_files = [f for f in os.listdir(self.db_path) if f.endswith(".pkl")]
        if not pkl_files:
            return

        pkl_path = os.path.join(self.db_path, pkl_files[0])
        pkl_time = os.path.getmtime(pkl_path)

        for root, dirs, files in os.walk(self.db_path):
            if os.path.getmtime(root) > pkl_time:
                for f in pkl_files:
                    try:
                        os.remove(os.path.join(self.db_path, f))
                        print(f"Cache invalidated (folder changed): {f}")
                    except Exception:
                        pass
                return
            for fname in files:
                if not fname.endswith(".pkl"):
                    fpath = os.path.join(root, fname)
                    if os.path.getmtime(fpath) > pkl_time:
                        for pf in pkl_files:
                            try:
                                os.remove(os.path.join(self.db_path, pf))
                                print(f"Cache invalidated (new file): {fname}")
                            except Exception:
                                pass
                        return

    def _worker(self):
        self._refresh_cache_if_needed()
        last_cache_check = time.time()

        while True:
            # Periodic live cache refresh (every 60 s)
            if time.time() - last_cache_check > 60:
                self._refresh_cache_if_needed()
                last_cache_check = time.time()

            try:
                track_id, face_crop = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                results = DeepFace.find(
                    img_path=face_crop,
                    db_path=self.db_path,
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                    distance_metric="cosine",
                    model_name="ArcFace",
                )

                MATCH_THRESHOLD = 0.50
                AMBIGUITY_RATIO = 0.08
                AMBIGUITY_FLOOR = 0.005

                person_name = "Guest"
                if results and not results[0].empty:
                    df          = results[0]
                    match       = df.iloc[0]
                    distance    = match["distance"]
                    identity    = match["identity"]
                    best_person = os.path.basename(os.path.dirname(identity))
                    print(f"ID {track_id}: {os.path.basename(identity)} | dist={distance:.4f}")

                    if distance < MATCH_THRESHOLD:
                        if len(df) >= 2:
                            second_identity = df.iloc[1]["identity"]
                            second_person   = os.path.basename(os.path.dirname(second_identity))
                            second_dist     = df.iloc[1]["distance"]
                            gap             = second_dist - distance

                            # FIX: If we have an excellent match (dist < 0.10), 
                            # don't let ambiguity block it.
                            if distance < 0.10 or best_person == second_person:
                                person_name = best_person.capitalize()
                            else:
                                min_gap = max(distance * AMBIGUITY_RATIO, AMBIGUITY_FLOOR)
                                if gap < min_gap:
                                    print(f"  Ambiguous (gap={gap:.4f} < min={min_gap:.4f}), marking Guest")
                                    person_name = "Guest"
                                else:
                                    person_name = best_person.capitalize()
                        else:
                            person_name = best_person.capitalize()
                    else:
                        print(f"  Distance {distance:.4f} > threshold, marking Guest")

                with self.lock:
                    self.results[track_id] = {
                        "name":      person_name,
                        "last_scan": time.time(),
                    }

            except Exception as e:
                print(f"Recognition error ID {track_id}: {e}")
                with self.lock:
                    existing = self.results.get(track_id, {})
                    self.results[track_id] = {
                        "name":      existing.get("name", "Guest"),
                        "last_scan": time.time(),
                    }
            finally:
                with self.lock:
                    self.processing.discard(track_id)

    def request(self, track_id: int, face_crop: np.ndarray):
        """Queue a recognition request; skip if already processing or recently scanned."""
        with self.lock:
            already_processing = track_id in self.processing
            last_scan = self.results.get(track_id, {}).get("last_scan", 0)
            recently_scanned = (time.time() - last_scan) < RESCAN_INTERVAL

        if already_processing or recently_scanned:
            return

        if not self.input_queue.full():
            with self.lock:
                self.processing.add(track_id)
            self.input_queue.put((track_id, face_crop))

    def get_name(self, track_id: int) -> str:
        with self.lock:
            return self.results.get(track_id, {}).get("name", "Scanning...")

    def clear(self, track_id: int):
        with self.lock:
            self.results.pop(track_id, None)
            self.processing.discard(track_id)

    def get_raw_name(self, track_id: int) -> str | None:
        """Return the latest raw recognition result, or None if not scanned yet."""
        with self.lock:
            entry = self.results.get(track_id)
            return entry["name"] if entry else None


recognizer = FaceRecognizer()

# ─── State ────────────────────────────────────────────────────────────────────
desk_timers      = {}
track_to_desk    = {}
person_positions = {}
person_velocity  = {}
person_prev_box  = {}
name_votes       = {} 
VOTE_THRESHOLD   = 3 
last_state       = None
last_update_time = 0.0

# ── Per-employee activity time tracking ───────────────────────────────────────
activity_start_time    = {}   # track_id → float  (when current activity started)
activity_accumulated   = {}   # track_id → {activity: total_seconds}
last_activity_by_track = {}   # track_id → last_activity_name

# ─── Helpers ──────────────────────────────────────────────────────────────────
ACTIVITY_COLORS = {
    "Working":      (0,   200,  50),
    "Using Mobile": (0,   0,   220),
    "Walking":      (220, 150,  0),
    "Sitting Idle": (0,   165, 255),
    "Standing":     (180,  0,  220),  
}

def get_face_crop(frame: np.ndarray, kpts: np.ndarray, box) -> np.ndarray | None:
    """
    Return a face crop. Strategy:
    1. Use nose + both eyes (ideal — frontal face)
    2. Fall back to nose + one eye (side-facing)
    3. Fall back to top portion of bounding box (last resort)
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = frame.shape[:2]
    nose  = kpts[0]
    l_eye = kpts[1]
    r_eye = kpts[2]

    anchor = None

    # Strategy 1: nose + both eyes visible
    if nose[2] > 0.5 and l_eye[2] > 0.5 and r_eye[2] > 0.5:
        anchor = nose
        side   = max(x2 - x1, y2 - y1) * 0.42

    # Strategy 2: nose + at least one eye (side-facing)
    elif nose[2] > 0.5 and (l_eye[2] > 0.5 or r_eye[2] > 0.5):
        anchor = nose
        side   = max(x2 - x1, y2 - y1) * 0.42

    # Strategy 3: use top 40% of bounding box as face region
    else:
        face_h = int((y2 - y1) * 0.45)
        crop = frame[
            max(0, y1):min(h, y1 + face_h),
            max(0, x1):min(w, x2),
        ]
        if crop.size > 0 and crop.shape[0] >= 60 and crop.shape[1] >= 60:
            return crop
        return None

    fx1 = int(anchor[0] - side)
    fy1 = int(anchor[1] - side * 1.3)
    fx2 = int(anchor[0] + side)
    fy2 = int(anchor[1] + side * 0.7)
    crop = frame[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
    if crop.size == 0 or crop.shape[0] < 60 or crop.shape[1] < 60:
        return None
    return crop


# ─── Main Loop ────────────────────────────────────────────────────────────────
TRACK_REUSE_JUMP_PX = 200

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        # Run trackers
        pose_results = pose_model.track(frame, persist=True, verbose=False,
                                        device="cpu", classes=[0],
                                        conf=0.5, iou=0.45)
        obj_results  = obj_model(frame, verbose=False, device="cpu", classes=[67])

        # Collect phone bounding boxes
        phone_boxes = []
        for r in obj_results:
            if r.boxes:
                for box in r.boxes.xyxy:
                    phone_boxes.append(box.cpu().numpy())

        desk_updates     = {}
        active_track_ids = set()

        if (pose_results
                and pose_results[0].boxes is not None
                and pose_results[0].boxes.id is not None):

            boxes     = pose_results[0].boxes.xyxy.cpu().numpy()
            track_ids = pose_results[0].boxes.id.int().cpu().tolist()
            kpts_data = pose_results[0].keypoints.data.cpu().numpy()

            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            # ── Filter 1: Remove tiny boxes ──────────────────────────────────
            # Lowered from 0.05 to 0.01 to detect far-away people
            MIN_BOX_RATIO = 0.01
            valid_indices = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i][:4]
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / frame_area >= MIN_BOX_RATIO:
                    valid_indices.append(i)

            # ── Filter 2: Remove duplicates by IoU overlap ───────────────────
            OVERLAP_THRESH = 0.30
            keep = []
            for i in valid_indices:
                dominated = False
                ax1, ay1, ax2, ay2 = boxes[i][:4]
                a_area = (ax2 - ax1) * (ay2 - ay1)
                for j in valid_indices:
                    if i == j:
                        continue
                    bx1, by1, bx2, by2 = boxes[j][:4]
                    b_area = (bx2 - bx1) * (by2 - by1)
                    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    iou   = inter / (a_area + b_area - inter + 1e-6)
                    if iou > OVERLAP_THRESH and a_area < b_area:
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)

            # ── Filter 3: Remove duplicates by center proximity ───────────────
            CENTER_DIST_THRESH = frame_w * 0.15
            final = []
            for i in keep:
                ax1, ay1, ax2, ay2 = boxes[i][:4]
                a_cx = (ax1 + ax2) / 2
                a_cy = (ay1 + ay2) / 2
                a_area = (ax2 - ax1) * (ay2 - ay1)
                dominated = False
                for j in keep:
                    if i == j:
                        continue
                    bx1, by1, bx2, by2 = boxes[j][:4]
                    b_cx = (bx1 + bx2) / 2
                    b_cy = (by1 + by2) / 2
                    b_area = (bx2 - bx1) * (by2 - by1)
                    dist = np.hypot(a_cx - b_cx, a_cy - b_cy)
                    if dist < CENTER_DIST_THRESH and a_area < b_area:
                        dominated = True
                        break
                if not dominated:
                    final.append(i)

            # Sort left-to-right for stable desk numbering
            sorted_indices = sorted(final, key=lambda k: boxes[k][0])

            for i in sorted_indices:
                track_id = track_ids[i]
                active_track_ids.add(track_id)
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box[:4])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # ── Detect track ID reuse ────────────────────────────────
                if track_id in person_prev_box:
                    prev_cx, prev_cy = person_prev_box[track_id]
                    jump = np.hypot(cx - prev_cx, cy - prev_cy)
                    if jump > TRACK_REUSE_JUMP_PX:
                        print(f"Track ID {track_id} reused (jump={jump:.0f}px) — resetting state.")
                        desk_timers.pop(track_id, None)
                        track_to_desk.pop(track_id, None)
                        person_positions.pop(track_id, None)
                        person_velocity.pop(track_id, None)
                        recognizer.clear(track_id)

                person_prev_box[track_id] = (cx, cy)

                # ── Stable Desk ID ────────────────────────────────────────────
                if track_id not in track_to_desk:
                    used = set(track_to_desk.values())
                    d    = 1
                    while f"D{d}" in used:
                        d += 1
                    track_to_desk[track_id] = f"D{d}"
                desk_id = track_to_desk[track_id]

                # ── Face Recognition + Vote Lock ──────────────────────────────
                kpts      = kpts_data[i]
                face_crop = get_face_crop(frame, kpts, box)
                if face_crop is not None:
                    recognizer.request(track_id, face_crop)

                raw_name = recognizer.get_raw_name(track_id)
                if raw_name is not None and raw_name not in ("Scanning...",):
                    vote = name_votes.get(track_id, {"name": raw_name, "count": 0})
                    if vote["name"] == raw_name:
                        vote["count"] = min(vote["count"] + 1, VOTE_THRESHOLD)
                    else:
                        # Different name detected — reset vote with new candidate
                        vote = {"name": raw_name, "count": 1}
                    name_votes[track_id] = vote

                # Use committed name only once vote threshold is reached
                committed = name_votes.get(track_id, {})
                if committed.get("count", 0) >= VOTE_THRESHOLD:
                    person_name = committed["name"]
                else:
                    person_name = recognizer.get_name(track_id)   # "Scanning..." until confident

                # ── Velocity Tracking ─────────────────────────────────────────
                if track_id not in person_positions:
                    person_positions[track_id] = (cx, cy)
                    person_velocity[track_id]  = 0.0
                else:
                    ox, oy = person_positions[track_id]
                    dist   = np.hypot(cx - ox, cy - oy)
                    if dist < 8:
                        dist = 0.0
                    # Heavy smoothing so velocity only rises with sustained movement
                    person_velocity[track_id] = (
                        person_velocity[track_id] * 0.85 + dist * 0.15
                    )
                    person_positions[track_id] = (cx, cy)

                velocity = person_velocity[track_id]

                # ── Activity Classification ───────────────────────────────────
                # YOLOv8-pose keypoint indices (COCO layout):
                #  0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear
                #  5=l_shoulder  6=r_shoulder
                #  7=l_elbow     8=r_elbow
                #  9=l_wrist    10=r_wrist
                # 11=l_hip      12=r_hip
                # 13=l_knee     14=r_knee
                # 15=l_ankle    16=r_ankle

                is_using_phone = any(
                    x1 <= (pb[0] + pb[2]) / 2 <= x2 and y1 <= (pb[1] + pb[3]) / 2 <= y2
                    for pb in phone_boxes
                )

                if is_using_phone:
                    activity = "Using Mobile"

                elif velocity > 20.0:
                    activity = "Walking"

                else:
                    box_h = y2 - y1  # pixel height of the bounding box
                    box_w = x2 - x1
                    aspect = box_h / (box_w + 1e-6)

                    lh_y, lh_c = kpts[11][1], kpts[11][2]   # left hip
                    rh_y, rh_c = kpts[12][1], kpts[12][2]   # right hip
                    la_y, la_c = kpts[15][1], kpts[15][2]   # left ankle
                    ra_y, ra_c = kpts[16][1], kpts[16][2]   # right ankle
                    lk_y, lk_c = kpts[13][1], kpts[13][2]   # left knee
                    rk_y, rk_c = kpts[14][1], kpts[14][2]   # right knee

                    hip_confidences = [(lh_y, lh_c), (rh_y, rh_c)]
                    visible_hips    = [(y, c) for y, c in hip_confidences if c > 0.4]

                    ankle_confidences = [(la_y, la_c), (ra_y, ra_c)]
                    visible_ankles    = [(y, c) for y, c in ankle_confidences if c > 0.4]

                    knee_confidences = [(lk_y, lk_c), (rk_y, rk_c)]
                    visible_knees    = [(y, c) for y, c in knee_confidences if c > 0.4]

                    is_standing = None   # None = undecided

                    if visible_hips:
                        avg_hip_y  = sum(y for y, _ in visible_hips) / len(visible_hips)
                        hip_ratio  = (avg_hip_y - y1) / (box_h + 1e-6)

                        if visible_ankles:

                            avg_ankle_y   = sum(y for y, _ in visible_ankles) / len(visible_ankles)
                            ankle_ratio   = (avg_ankle_y - y1) / (box_h + 1e-6)
                            leg_span      = ankle_ratio - hip_ratio  # large → legs extended → standing

                            # Standing when hips are mid-box AND legs are clearly extended
                            is_standing = (hip_ratio > 0.35) and (leg_span > 0.30)

                        elif visible_knees:
                            # Strategy 1b — hips + knees but no ankles (common for seated
                            # people whose feet are under a desk).
                            avg_knee_y = sum(y for y, _ in visible_knees) / len(visible_knees)
                            knee_ratio = (avg_knee_y - y1) / (box_h + 1e-6)
                            knee_span  = knee_ratio - hip_ratio

                            is_standing = (hip_ratio > 0.35) and (knee_span > 0.20)

                        else:
                            is_standing = hip_ratio > 0.42

                    if is_standing is None:
                        # Tall narrow box → standing; short wide box → sitting.
                        is_standing = aspect > 1.6

                    # ── Classify sitting vs standing ──────────────────────────
                    lw_y, lw_c = kpts[9][1],  kpts[9][2]   # left wrist
                    rw_y, rw_c = kpts[10][1], kpts[10][2]  # right wrist
                    ls_y, rs_y = kpts[5][1],  kpts[6][1]   # shoulders
                    lh_y, rh_y = kpts[11][1], kpts[12][1]  # hips
                    le_y, re_y = kpts[7][1],  kpts[8][1]   # elbows

                    if is_standing:

                        mid_y = (ls_y + lh_y) / 2
                        hands_raised = (
                            (lw_c > 0.5 and lw_y < mid_y) or
                            (rw_c > 0.5 and rw_y < ((rs_y + rh_y) / 2))
                        )
                        activity = "Working" if hands_raised else "Standing"
                    else:
                        # While sitting, working = wrists between shoulders and hips
                        hands_active = (
                            (lw_c > 0.5 and ls_y < lw_y < lh_y) or
                            (rw_c > 0.5 and rs_y < rw_y < rh_y)
                        )
                        activity = "Working" if hands_active else "Sitting Idle"

                # ── Activity Time Accumulator ──────────────────────────────────
                if track_id not in activity_start_time:
                    activity_start_time[track_id]    = current_time
                    activity_accumulated[track_id]   = {}
                    last_activity_by_track[track_id] = activity
                else:
                    prev_activity = last_activity_by_track.get(track_id, activity)
                    if prev_activity != activity:
                        elapsed = current_time - activity_start_time[track_id]
                        activity_accumulated[track_id][prev_activity] = (
                            activity_accumulated[track_id].get(prev_activity, 0) + elapsed
                        )
                        activity_start_time[track_id] = current_time
                    last_activity_by_track[track_id] = activity

                # ── Timer ─────────────────────────────────────────────────────
                if track_id not in desk_timers:
                    desk_timers[track_id] = current_time
                duration = int(current_time - desk_timers[track_id])

                # ── Draw ──────────────────────────────────────────────────────
                color = ACTIVITY_COLORS.get(activity, (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{person_name} ({desk_id}) | {activity} | {duration}s"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                desk_updates[desk_id] = {
                    "status":      "Occupied",
                    "activity":    activity,
                    "person_name": person_name,
                    "time":        duration,
                }

                # ── Build activity-log snapshot for known employees only ────────
                if person_name not in ("Scanning...", "Guest", "—", ""):
                    ongoing = current_time - activity_start_time.get(track_id, current_time)
                    snap    = dict(activity_accumulated.get(track_id, {}))
                    snap[activity] = snap.get(activity, 0) + ongoing
                    if "_activity_log" not in desk_updates:
                        desk_updates["_activity_log"] = {}
                    desk_updates["_activity_log"][person_name] = {
                        k: round(v) for k, v in snap.items() if v > 0
                    }

        # ── Cleanup stale tracks ───────────────────────────────────────────────
        for t_id in list(desk_timers.keys()):
            if t_id not in active_track_ids:
                desk_timers.pop(t_id, None)
                track_to_desk.pop(t_id, None)
                person_positions.pop(t_id, None)
                person_velocity.pop(t_id, None)
                person_prev_box.pop(t_id, None)
                name_votes.pop(t_id, None)
                activity_start_time.pop(t_id, None)
                activity_accumulated.pop(t_id, None)
                last_activity_by_track.pop(t_id, None)
                recognizer.clear(t_id)

        # ── Send to Backend ────────────────────────────────────────────────────
        # Extract activity log before comparing desk state
        activity_log_payload = desk_updates.pop("_activity_log", {})

        if desk_updates != last_state or (current_time - last_update_time) > UPDATE_INTERVAL:
            try:
                payload = [{"desk_id": d, **v} for d, v in desk_updates.items()]
                requests.post(BACKEND_URL, json=payload, timeout=0.3)
                
                # Send activity log to backend synchronized with desk updates
                if activity_log_payload:
                    requests.post(
                        "http://localhost:8000/activity_log",
                        json=activity_log_payload,
                        timeout=0.3,
                    )
                
                last_state       = desk_updates.copy()
                last_update_time = current_time
            except requests.exceptions.RequestException:
                pass

        cv2.imshow("Smart Office Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")