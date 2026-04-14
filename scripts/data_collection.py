import cv2
import os
import numpy as np
import albumentations as A
from retinaface import RetinaFace
from deepface import DeepFace
import requests

# --- 1. CONFIGURATION ---
VIDEO_PATH = "F:/attendance_project_material/VID20260408113027.mp4"
STUDENT_NAME = "Prajwal_Pawar"
ENROLLMENT_NUM = "103223"
OUTPUT_FOLDER = f"F:/class_attendance/dataset/{STUDENT_NAME}"

BLUR_THRESHOLD = 10.0
FRAME_SKIP = 10

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- 2. ALBUMENTATIONS PIPELINE ---
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
])

# --- 3. CLEANING HELPER FUNCTIONS ---
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def get_image_hash(image):
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    return np.where(gray > avg, 1, 0).flatten()

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

# --- 4. MAIN VIDEO PROCESSING LOOP ---
def process_video():
    vid = cv2.VideoCapture(VIDEO_PATH)

    # ── DIAGNOSTIC: Print video properties first ──────────────────────────────
    if not vid.isOpened():
        print("❌ FATAL: Could not open video file. Check the path.")
        return 0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = vid.get(cv2.CAP_PROP_FPS)
    width        = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Video Info → Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    print(f"   With FRAME_SKIP={FRAME_SKIP}, we will sample ~{total_frames // FRAME_SKIP} candidate frames.\n")
    # ──────────────────────────────────────────────────────────────────────────

    count = 0
    saved_count = 0
    hash_history = []

    # ── DIAGNOSTIC: Counters for each filter stage ────────────────────────────
    total_candidates  = 0   # frames that reached processing (passed FRAME_SKIP)
    dropped_blur      = 0
    dropped_duplicate = 0
    dropped_retinaface_error = 0
    dropped_no_face   = 0
    # ──────────────────────────────────────────────────────────────────────────

    print(f"Starting extraction for {STUDENT_NAME}...")

    while True:
        success, frame = vid.read()
        if not success:
            break

        if count % FRAME_SKIP == 0:
            total_candidates += 1

            # Check 1: Blur Filtering
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < BLUR_THRESHOLD:
                # ── DIAGNOSTIC: Print blur score so you can tune the threshold
                print(f"  [BLUR] Frame {count} dropped — variance={variance:.2f} (threshold={BLUR_THRESHOLD})")
                dropped_blur += 1
                count += 1
                continue

            # Check 2: Duplicate Frame Filtering
            current_hash = get_image_hash(frame)
            is_duplicate = False
            for prev_hash in hash_history:
                if hamming_distance(current_hash, prev_hash) <= 2:
                    is_duplicate = True
                    break

            if is_duplicate:
                dropped_duplicate += 1
                count += 1
                continue

            hash_history.append(current_hash)

            # Check 3: Face Detection & Alignment using RetinaFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                detected_faces = RetinaFace.extract_faces(img_path=rgb_frame, align=True)
            except Exception as e:
                # ── DIAGNOSTIC: Print the actual error instead of silently skipping
                print(f"  [RETINAFACE ERROR] Frame {count}: {e}")
                dropped_retinaface_error += 1
                count += 1
                continue

            if len(detected_faces) == 0:
                print(f"  [NO FACE] Frame {count} — RetinaFace found 0 faces.")
                dropped_no_face += 1
                count += 1
                continue

            aligned_face = detected_faces[0]
            bgr_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            bgr_face = cv2.resize(bgr_face, (112, 112))

            base_filename = os.path.join(OUTPUT_FOLDER, f"clean_{saved_count}.jpg")
            cv2.imwrite(base_filename, bgr_face)

            for aug_idx in range(2):
                augmented = augmentor(image=bgr_face)['image']
                aug_filename = os.path.join(OUTPUT_FOLDER, f"aug_{saved_count}_{aug_idx}.jpg")
                cv2.imwrite(aug_filename, augmented)

            print(f"  [SAVED] Frame {count} → clean_{saved_count}.jpg (blur variance={variance:.2f})")
            saved_count += 1

        count += 1

    vid.release()

    # ── DIAGNOSTIC: Full summary breakdown ────────────────────────────────────
    print("\n" + "="*50)
    print(f"  EXTRACTION SUMMARY for {STUDENT_NAME}")
    print("="*50)
    print(f"  Total frames in video   : {total_frames}")
    print(f"  Candidate frames sampled: {total_candidates}")
    print(f"  Dropped — blurry        : {dropped_blur}")
    print(f"  Dropped — duplicate     : {dropped_duplicate}")
    print(f"  Dropped — RetinaFace err: {dropped_retinaface_error}")
    print(f"  Dropped — no face found : {dropped_no_face}")
    print(f"  ✅ Saved (clean)        : {saved_count}")
    print(f"  ✅ Saved (augmented)    : {saved_count * 2}")
    print("="*50 + "\n")
    # ──────────────────────────────────────────────────────────────────────────

    return saved_count

# --- 5. EMBEDDING GENERATION ---
def generate_embeddings():
    print("Generating ArcFace Embeddings...")
    embeddings_list = []
    filenames_list = []

    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".jpg"):
            img_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                embedding_objs = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
                raw_embedding = embedding_objs[0]["embedding"]
                vector = np.array(raw_embedding)
                normalized_vector = vector / np.linalg.norm(vector)
                embeddings_list.append(normalized_vector.tolist())
                filenames_list.append(filename)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    
      # ── EMBEDDING LIST DISPLAY ────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  EMBEDDING SUMMARY  ({len(embeddings_list)} total)")
    print("="*50)
    for i, (fname, emb) in enumerate(zip(filenames_list, embeddings_list)):
        emb_array = np.array(emb)
        print(f"\n  [{i+1:>3}] File       : {fname}")
        print(f"        Dimensions : {len(emb)}")
        print(f"        L2 norm    : {np.linalg.norm(emb_array):.6f}  (should be ~1.0)")
        print(f"        Min / Max  : {emb_array.min():.6f} / {emb_array.max():.6f}")
        print(f"        First 8    : {np.round(emb_array[:8], 6).tolist()}")
    print("\n" + "="*50)
 
    # Full raw dump — useful if you want to copy-paste into a DB or JSON file
    print("\n📋 FIRST 5 EMBEDDING LIST (normalized vectors):\n")
    for i, (fname, emb) in enumerate(zip(filenames_list[:1], embeddings_list[:1])):
        print(f"  # {fname}")
        print(f"  {emb}\n")
    # ──────────────────────────────────────────────────────────────────────────

    print(f"Successfully generated {len(embeddings_list)} normalized embeddings.")
    return embeddings_list, filenames_list

if __name__ == "__main__":
    frames_extracted = process_video()
    if frames_extracted > 0:
        embeddings, filenames = generate_embeddings()
        print(f"\nPipeline ready! We have {len(embeddings)} arrays ready for the database.")
        
        # --- NEW INTEGRATION STEP ---
        print("\nUploading data to the database...")
        payload = {
            "name": STUDENT_NAME,
            "enrollment_number": ENROLLMENT_NUM,
            "face_embeddings": embeddings
        }
        
        # Ensure your FastAPI server is running before executing this!
        response = requests.post("http://127.0.0.1:8000/register/", json=payload)
        
        if response.status_code == 200:
            print("✅ Success:", response.json())
        elif response.status_code == 400:
            print("⚠️  Duplicate student:", response.json()["detail"])
        else:
            print(f"❌ Failed ({response.status_code}):", response.text)    
    else:
        print("\n⚠️  No frames were saved — embeddings skipped. Fix the extraction stage first.")