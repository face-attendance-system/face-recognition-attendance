import os
import tempfile
import cv2
import numpy as np
import albumentations as A
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from retinaface import RetinaFace
from deepface import DeepFace
from fastapi.responses import FileResponse

# Import your local database and models
import models
from database import engine, Base, get_db

app = FastAPI(title="Automated Attendance Portal")

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
                is_duplicate = any(hamming_distance(current_hash, prev_hash) <= 2 for prev_hash in hash_history)
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