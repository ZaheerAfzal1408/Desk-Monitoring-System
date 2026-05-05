import cv2
import os

# Use OpenCV's built-in face detector to validate frames during enrollment
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
MIN_PHOTOS    = 3   # Minimum required before quitting
TARGET_PHOTOS = 5   # Recommended target shown to user


def detect_face(frame) -> tuple[bool, tuple]:
    """
    Returns (face_found, (fx, fy, fw, fh)).
    Validates that the frame actually contains a face before saving,
    preventing bad/blank captures from poisoning the recognition database.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return False, ()
    # Return the largest detected face
    largest = max(faces, key=lambda f: f[2] * f[3])
    return True, tuple(largest)


def enroll():
    os.makedirs("employees", exist_ok=True)

    name = input("Enter the employee's name: ").strip()
    if not name:
        print("Error: Name cannot be empty.")
        return

    person_dir = os.path.join("employees", name.replace(" ", "_"))

    # ── FIX: Warn when adding to an existing person ───────────────────────────
    # Re-enrolling appends photos (which is useful), but the user should know
    # they are adding to an existing record rather than creating a fresh one.
    existing_photos = 0
    if os.path.isdir(person_dir):
        existing_photos = len([
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if existing_photos > 0:
            print(f"\n⚠  '{name}' already has {existing_photos} photo(s) enrolled.")
            choice = input("  (A)ppend more photos, (R)eplace all, or (C)ancel? [A/R/C]: ").strip().upper()
            if choice == "C":
                print("Enrollment cancelled.")
                return
            elif choice == "R":
                # Remove existing photos and the pkl cache so DB rebuilds
                for f in os.listdir(person_dir):
                    fpath = os.path.join(person_dir, f)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                # Wipe pkl cache so DeepFace rebuilds from scratch
                db_root = "employees"
                for f in os.listdir(db_root):
                    if f.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(db_root, f))
                        except Exception:
                            pass
                print(f"  Existing photos removed. Starting fresh enrollment for '{name}'.")
                existing_photos = 0
            else:
                print(f"  Appending to existing {existing_photos} photo(s).")

    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count        = 0
    no_face_warn = False   # Debounce the "no face" warning

    print(f"\n─── Enrolling: {name} ───────────────────────────────────────")
    print(f"  Take {TARGET_PHOTOS} photos from slightly different angles.")
    print("  S = Save photo (only saved if a face is detected)")
    print("  Q = Finish enrollment")
    print("────────────────────────────────────────────────────────────\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera frame read failed.")
            break

        # Always save the original (un-mirrored) frame so DeepFace
        # sees the same orientation as main.py captures.
        display = cv2.flip(frame, 1)   # Mirrored for comfortable self-view only

        face_found, face_rect = detect_face(frame)

        # Draw guide box around detected face on the display frame
        if face_found:
            fx, fy, fw, fh = face_rect
            # Mirror the x-coordinate for display only
            display_fx = display.shape[1] - fx - fw
            cv2.rectangle(display, (display_fx, fy),
                          (display_fx + fw, fy + fh), (0, 220, 80), 2)
            no_face_warn = False

        # Status overlay
        bar_color = (0, 180, 60) if face_found else (0, 80, 200)
        cv2.rectangle(display, (0, 0), (display.shape[1], 70), (20, 20, 30), -1)

        status = "Face detected ✓" if face_found else "No face detected"
        cv2.putText(display, f"Enrolling: {name}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(display, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

        # Progress bar
        total_target = TARGET_PHOTOS  # count is session-only; bar reflects session progress
        progress = min(count / total_target, 1.0)
        bar_w    = int(display.shape[1] * progress)
        cv2.rectangle(display, (0, display.shape[0] - 8),
                      (bar_w, display.shape[0]), bar_color, -1)

        # FIX: Show total (including pre-existing) in the counter so the user
        # understands how many photos are in the database for this person.
        total_saved  = existing_photos + count
        counter_text = (
            f"Session: {count}/{TARGET_PHOTOS}  |  Total in DB: {total_saved}"
            f"  |  S=Save  Q=Quit"
        )
        cv2.putText(display, counter_text,
                    (10, display.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.imshow(f"Enroll — {name}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if not face_found:
                print("⚠  No face detected — move closer or improve lighting.")
                no_face_warn = True
            else:
                # Save the original un-mirrored frame
                # Use a unique timestamp suffix to avoid collisions when appending
                ts       = int(time.time() * 1000) if count == 0 else count
                filename = os.path.join(
                    person_dir,
                    f"{name.replace(' ', '_')}_{existing_photos + count:03d}.jpg"
                )
                cv2.imwrite(filename, frame)
                count += 1
                print(f"[{count}/{TARGET_PHOTOS}] Saved → {filename}")

        elif key == ord("q"):
            # ── FIX: Warn about low photo count even when confirmed ────────────
            # A 1- or 2-photo enrollment stays on disk without warning after
            # the old code; now we always show the risk and log it clearly.
            total_in_db = existing_photos + count
            if total_in_db < MIN_PHOTOS:
                print(f"\n⚠  Only {total_in_db} photo(s) total for '{name}' "
                      f"(minimum recommended: {MIN_PHOTOS}).")
                print("   Recognition accuracy will be poor with fewer than "
                      f"{MIN_PHOTOS} photos.")
                confirm = input("Quit anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        total_in_db = existing_photos + count
        print(f"\n✓ Enrollment complete: {count} new photo(s) saved for '{name}'.")
        print(f"  Total photos in DB: {total_in_db}")
        print(f"  Folder: {person_dir}")
        print("  Run main.py — the recognition database will update automatically.")
    else:
        print("\nNo photos saved. Enrollment cancelled.")
        # Only remove the folder if it was freshly created and is now empty
        try:
            if not os.listdir(person_dir):
                os.rmdir(person_dir)
        except OSError:
            pass


# ── Import time only when needed (enrollment is rarely run from main.py) ──────
import time

if __name__ == "__main__":
    enroll()