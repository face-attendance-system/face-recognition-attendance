import os
import tempfile
import cv2
import numpy as np
import albumentations as A
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import func
from retinaface import RetinaFace
from deepface import DeepFace
from retinaface.commons import postprocess
from fastapi.responses import FileResponse
import math
import uuid
from fastapi.staticfiles import StaticFiles
import schemas
from scipy.optimize import linear_sum_assignment
import scipy.spatial
import json
# Import your local database and models
import models
from database import engine, Base, get_db

app = FastAPI(title="Automated Attendance Portal")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    # This assumes index.html is in the same folder as main.py
    return FileResponse("static/index.html")

# Allow your frontend HTML file to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CONFIGURATION & HELPERS ---
BLUR_THRESHOLD = 10.0
FRAME_SKIP = 10

augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
])


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
    return np.sum(hash1!= hash2)

# --- 2. STARTUP EVENT ---
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- 3. THE REGISTRATION ENDPOINT ---
@app.post("/register-video/")
async def register_student_video(
    name: str = Form(...),
    enrollment_number: str = Form(...),
    video_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # 3a. Check if student already exists in the database
    result = await db.execute(select(models.Student).where(models.Student.enrollment_number == enrollment_number))
    existing_student = result.scalars().first()
    if existing_student:
        raise HTTPException(status_code=400, detail="Student with this enrollment number already exists.")

    # 3b. Save uploaded video to a temporary file (Required for OpenCV)
    try:
        # delete=False is used to prevent Windows permission errors while OpenCV is reading it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await video_file.read())
            temp_video_path = temp_video.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    # 3c. Process the video and generate embeddings
    embeddings_list = []
    hash_history = []
    count = 0

    # ── DIAGNOSTIC: Counters for each filter stage ────────────────────────────
    total_candidates  = 0   # frames that reached processing (passed FRAME_SKIP)
    dropped_blur      = 0
    dropped_duplicate = 0
    dropped_retinaface_error = 0
    dropped_no_face   = 0
    dropped_embedding_error = 0
    clean_faces_saved = 0
    # ──────────────────────────────────────────────────────────────────────────
    
    try:
        vid = cv2.VideoCapture(temp_video_path)
        while True:
            success, frame = vid.read()
            if not success:
                break

            if count % FRAME_SKIP == 0:
                total_candidates += 1
                # Filter Blurry
                if is_blurry(frame):
                    dropped_blur += 1
                    count += 1
                    continue

                # Filter Duplicates
                current_hash = get_image_hash(frame)
                is_duplicate = any(hamming_distance(current_hash, prev_hash) <= 0 for prev_hash in hash_history)
                if is_duplicate:
                    dropped_duplicate += 1
                    count += 1
                    continue

                hash_history.append(current_hash)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect and Align with RetinaFace
                try:
                    detected_faces = RetinaFace.extract_faces(img_path=rgb_frame, align=True)
                except Exception:
                    dropped_retinaface_error += 1
                    count += 1
                    continue

                if len(detected_faces)!= 1:
                    dropped_no_face += 1
                    count += 1
                    continue

                aligned_face = detected_faces[0]
                bgr_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                bgr_face = cv2.resize(bgr_face, (112, 112))

                # Augmentation Array
                faces_to_embed = [bgr_face]
                clean_faces_saved += 1

                for _ in range(2):
                    augmented = augmentor(image=bgr_face)['image']
                    faces_to_embed.append(augmented)

                # Extract Embeddings directly from memory array
                for face_img in faces_to_embed:
                    try:
                        # DeepFace accepts NumPy arrays directly, eliminating disk write/read time
                        embedding_objs = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)
                        raw_embedding = embedding_objs[0]["embedding"]
                        
                        vector = np.array(raw_embedding)
                        normalized_vector = vector / np.linalg.norm(vector)
                        embeddings_list.append(normalized_vector.tolist())
                    except Exception:
                        dropped_embedding_error += 1

            count += 1
        vid.release()
    finally:
        # 3d. Always clean up and delete the temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # 3e. Failsafe if video was completely blurry or invalid
    if not embeddings_list:
        raise HTTPException(status_code=400, detail="Could not extract any valid face embeddings from the video.")

    # 3f. Save to PostgreSQL Database
    new_student = models.Student(name=name, enrollment_number=enrollment_number)
    db.add(new_student)
    await db.commit()
    await db.refresh(new_student)

    embeddings_to_insert = [
    models.StudentEmbedding(
        student_id=new_student.id,
        embedding=emb
    )
    for emb in embeddings_list
]
    db.add_all(embeddings_to_insert)
    await db.commit()

    return {
    "status": "success",
    "name": new_student.name,
    "total_embeddings_saved": len(embeddings_to_insert),
    "summary": {
        "total_candidates_sampled" : total_candidates,
        "dropped_blurry"           : dropped_blur,
        "dropped_duplicate"        : dropped_duplicate,
        "dropped_retinaface_error" : dropped_retinaface_error,
        "dropped_no_face"          : dropped_no_face,
        "dropped_embedding_error"  : dropped_embedding_error,
        "clean_faces_extracted"    : total_candidates - dropped_blur - dropped_duplicate - dropped_retinaface_error - dropped_no_face,
        "total_embeddings_saved"   : len(embeddings_to_insert),
        "augmented_faces_generated" : clean_faces_saved * 2
    }
}

def align_and_crop(img_large, face_area, landmarks):
    lx1, ly1, lx2, ly2 = face_area
    orig_w = int(lx2 - lx1)
    orig_h = int(ly2 - ly1)

    # 1. Pad by 50% to prevent corner clipping during rotation
    pad_w = int(orig_w * 0.5)
    pad_h = int(orig_h * 0.5)

    crop_x1 = max(0, int(lx1 - pad_w))
    crop_y1 = max(0, int(ly1 - pad_h))
    crop_x2 = min(img_large.shape[1], int(lx2 + pad_w))
    crop_y2 = min(img_large.shape[0], int(ly2 + pad_h))          # FIX 1: shape[0]

    cropped = img_large[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if cropped.shape[0] < 10 or cropped.shape[1] < 10:            # FIX 2: shape[0]
        return None

    # 2. Localize landmarks relative to the padded crop
    visual_left_eye  = landmarks["right_eye"]
    visual_right_eye = landmarks["left_eye"]

    left_eye  = (visual_left_eye[0]  - crop_x1, visual_left_eye[1]  - crop_y1)   # FIX 3
    right_eye = (visual_right_eye[0] - crop_x1, visual_right_eye[1] - crop_y1)   # FIX 4

    # 3. Calculate angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]                               # FIX 5: [0]
    angle = math.degrees(math.atan2(dy, dx))

    # 4. Find the Eye Center
    eye_center = (
        int((left_eye[0] + right_eye[0]) / 2),                   # FIX 6: [0]
        int((left_eye[1] + right_eye[1]) / 2)
    )

    # 5. Rotate the padded image
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(
        cropped, M,
        (cropped.shape[1], cropped.shape[0]),                     # FIX 7: shape[0]
        flags=cv2.INTER_CUBIC
    )

    # 6. Re-crop tightly, anchored around the eye center
    new_x1 = max(0, eye_center[0] - orig_w // 2)                 # FIX 8: eye_center[0]
    new_x2 = min(rotated.shape[1], eye_center[0] + orig_w // 2)  # FIX 9: eye_center[0]
    new_y1 = max(0, eye_center[1] - int(orig_h * 0.4))
    new_y2 = min(rotated.shape[0], eye_center[1] + int(orig_h * 0.6))  # FIX 10: shape[0]

    final_face = rotated[new_y1:new_y2, new_x1:new_x2]

    # Failsafe resize
    if final_face.shape[0] < 10 or final_face.shape[1] < 10:     # FIX 11: shape[0]
        final_face = cv2.resize(cropped, (112, 112))
    else:
        final_face = cv2.resize(final_face, (112, 112))

    # 7. Apply CLAHE to restore contrast on small back-row faces
    lab = cv2.cvtColor(final_face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_face = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return final_face


@app.post("/analyze-group-photo/")
async def analyze_group_photo(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    os.makedirs("static/proofs", exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        temp_img.write(await file.read())
        temp_img_path = temp_img.name

    try:
        # 1. Fetch and aggregate embeddings from PostgreSQL using pgvector's AVG()
        stmt = (
            select(
                models.Student.id,
                models.Student.name,
                models.Student.enrollment_number,
                func.avg(models.StudentEmbedding.embedding).label("avg_embedding")
            )
          .join(models.StudentEmbedding, models.Student.id == models.StudentEmbedding.student_id)
          .group_by(models.Student.id)
        )
        
        result = await db.execute(stmt)
        aggregated_students = result.all()

        unique_students_ids = []
        unique_students_vectors = []
        checklist = {}

        # 2. Parse the vectors and build the UI checklist
        for row in aggregated_students:
            unique_students_ids.append(row.id)
            
            # Not used -> Use ast.literal_eval to safely parse the string back into a numerical array
            avg_vec = np.array(json.loads(row.avg_embedding)) if isinstance(row.avg_embedding, str) else np.array(row.avg_embedding)
            normalized_avg_vec = avg_vec / np.linalg.norm(avg_vec)
            unique_students_vectors.append(normalized_avg_vec)
            
            checklist[row.id] = {
                "student_id": row.id,
                "name": row.name,
                "enrollment_number": row.enrollment_number,
                "is_present": False,
                "confidence_score": 0.0
            }

        # 3. Read image and UPSCALE to catch small/occluded faces in the back rows
        img = cv2.imread(temp_img_path)
        scale_factor = 2.5
        img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        temp_large_path = f"temp_large_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_large_path, img_large)

        detected_faces = RetinaFace.detect_faces(temp_large_path, threshold=0.5)
        os.remove(temp_large_path)

        # 4. Unified drawing list and embedding extraction
        face_boxes = []
        face_embeddings = []
        box_indices = [] # Maps the embedding index back to the face_boxes list

        if isinstance(detected_faces, dict):
            for key, face_data in detected_faces.items():
                lx1, ly1, lx2, ly2 = face_data["facial_area"]
                
                # Scale coordinates back down to the original image size for drawing
                x1, y1 = int(lx1 / scale_factor), int(ly1 / scale_factor)
                x2, y2 = int(lx2 / scale_factor), int(ly2 / scale_factor)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                box_idx = len(face_boxes)
                face_boxes.append({
                    "coords": (x1, y1, x2, y2),
                    "label": "Unknown",
                    "color": (0, 0, 255) 
                })
                
                # Prevent out-of-bounds cropping on the large image
                lx1, ly1 = max(0, lx1), max(0, ly1)
                lx2, ly2 = min(img_large.shape[1], lx2), min(img_large.shape[0], ly2)
                
                cropped_bgr = img_large[ly1:ly2, lx1:lx2].copy()
                
                if cropped_bgr.shape[0] < 10 or cropped_bgr.shape[1] < 10:
                    continue
                
                # --- THE FIX: DYNAMIC FACIAL ALIGNMENT ---
                landmarks = face_data["landmarks"]
                
                # Map anatomical eyes to visual eyes
                visual_left_eye = landmarks["right_eye"]
                visual_right_eye = landmarks["left_eye"]
                
                # Shift landmarks to the local coordinates of the clamped crop
                left_eye_local = (visual_left_eye[0] - lx1, visual_left_eye[1] - ly1)
                right_eye_local = (visual_right_eye[0] - lx1, visual_right_eye[1] - ly1)
                
                try:
                    # Use our custom function that safely returns a pure NumPy array
                    aligned_bgr = align_cropped_face(cropped_bgr, left_eye_local, right_eye_local)
                except Exception:
                    aligned_bgr = cropped_bgr # Fallback only if calculation fails
                        
                # Standardize resolution for ArcFace
                aligned_bgr = cv2.resize(aligned_bgr, (112, 112))

                try:
                    # Pass the perfectly ALIGNED face to DeepFace
                    emb_objs = DeepFace.represent(img_path=aligned_bgr, model_name="ArcFace", enforce_detection=False)
                    vector = np.array(emb_objs[0]["embedding"])
                    normalized_vector = vector / np.linalg.norm(vector)
                    
                    face_embeddings.append(normalized_vector)
                    box_indices.append(box_idx)
                except Exception:
                    continue


# 5. Build Cost Matrix and run Hungarian Algorithm
        if len(face_embeddings) > 0 and len(unique_students_vectors) > 0:
            # Define your minimum confidence threshold
            MIN_CONFIDENCE = 68.0 
            
            # Convert confidence percentage back to cosine distance for the matrix
            min_angle = (1.0 - (MIN_CONFIDENCE / 100.0)) * math.pi
            threshold_distance = 1.0 - math.cos(min_angle)

            # Create the real cost matrix
            cost_matrix = np.zeros((len(face_embeddings), len(unique_students_vectors)))
            for i, face_vec in enumerate(face_embeddings):
                for j, student_vec in enumerate(unique_students_vectors):
                    cost_matrix[i, j] = scipy.spatial.distance.cosine(face_vec, student_vec)

            # THE FIX: Create a Dummy Matrix and pad the original matrix
            # We add N dummy columns, each with a fixed distance cost equal to our threshold
            dummy_matrix = np.full((len(face_embeddings), len(face_embeddings)), threshold_distance)
            padded_cost_matrix = np.hstack((cost_matrix, dummy_matrix))

            # Run Hungarian algorithm on the padded matrix
            row_indices, col_indices = linear_sum_assignment(padded_cost_matrix)

            # 6. Process Optimal Matches and UPGRADE to Green Boxes
            for row_idx, col_idx in zip(row_indices, col_indices):
                # Check if the algorithm assigned the face to a REAL student or a DUMMY
                if col_idx < len(unique_students_ids):
                    distance = padded_cost_matrix[row_idx, col_idx]
                    student_id = unique_students_ids[col_idx]
                    actual_box_idx = box_indices[row_idx]
                    
                    cos_sim = max(-1.0, min(1.0, 1.0 - distance)) 
                    angle = math.acos(cos_sim)
                    confidence = (1.0 - (angle / math.pi)) * 100.0

                    student_name = checklist[student_id]["name"]
                    print(f"Face {row_idx} → {student_name}: confidence={confidence:.2f}%")

                    # Double check the confidence passes the threshold
                    if confidence >= MIN_CONFIDENCE:
                        student_info = checklist[student_id]
                        student_info["is_present"] = True
                        student_info["confidence_score"] = round(confidence, 2)
                        
                        face_boxes[actual_box_idx]["label"] = f"{student_info['name']} ({confidence:.1f}%)"
                        face_boxes[actual_box_idx]["color"] = (0, 255, 0)

        # 7. Draw ALL boxes cleanly in one single pass
        for box in face_boxes:
            x1, y1, x2, y2 = box["coords"]
            color = box["color"]
            label = box["label"]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 8. Save and Return
        proof_filename = f"session_{uuid.uuid4().hex}.jpg"
        proof_filepath = os.path.join("static/proofs", proof_filename)
        cv2.imwrite(proof_filepath, img)
        
        proof_url = f"http://127.0.0.1:8000/static/proofs/{proof_filename}"

        return {
            "status": "success",
            "proof_image_url": proof_url,
            "checklist": list(checklist.values())
        }
        
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@app.post("/submit-attendance/")
async def submit_attendance(data: schemas.AttendanceSubmit, db: AsyncSession = Depends(get_db)):
    # 1. Create a new Attendance Session (The Lecture)
    new_session = models.AttendanceSession(
        subject_name=data.subject_name,
        proof_image_path=data.proof_image_url  # Storing the URL/Path to the saved image
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session) # Get the generated session_id

    # 2. Create the individual Student Logs
    logs_to_insert = []
    for record in data.records:
        new_log = models.AttendanceLog(
            session_id=new_session.id,
            student_id=record.student_id,
            is_present=record.is_present,
            confidence_score=record.confidence_score
        )
        logs_to_insert.append(new_log)

    db.add_all(logs_to_insert)
    await db.commit()

    return {"status": "success", "message": "Attendance successfully recorded.", "session_id": new_session.id}