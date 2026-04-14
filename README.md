# Face Recognition Attendance System – Stage 3

A smart AI-powered attendance system that uses Face Recognition to automatically detect and mark student attendance from group images.

## Stage 3 introduces a complete end-to-end attendance pipeline:
	•	👤 Register students using video (multiple embeddings)
	•	📸 Upload classroom/group image
	•	🧠 Detect & recognize faces using AI
	•	✅ Generate attendance checklist
	•	📝 Teacher verification
	•	💾 Store final attendance in database
  
  ## Key Features

### Face Recognition Pipeline
	•	Face Detection using RetinaFace
	•	Embedding extraction using DeepFace (ArcFace)
	•	Vector similarity matching using Cosine Distance
	•	Hungarian Algorithm for optimal matching


### Smart Matching System
	•	Uses pgvector + PostgreSQL
	•	Stores multiple embeddings per student
	•	Aggregates embeddings using AVG() for stability
	•	Confidence-based recognition

### Attendance Workflow
	1.	Upload group photo
	2.	AI detects faces
	3.	Matches with database
	4.	Generates checklist
	5.	Teacher verifies
	6.	Final attendance stored


### Visualization
	•	Bounding boxes on detected faces
	•	Green → Recognized
	•	Red → Unknown
	•	Confidence score displayed

### Tech Stack
    Backend
	•	FastAPI
	•	SQLAlchemy (Async)
	•	PostgreSQL
	•	pgvector

### AI / Computer Vision
	•	OpenCV
	•	DeepFace (ArcFace)
	•	RetinaFace
	•	Albumentations

### Algorithms
	•	Cosine Similarity

 ### Project Structure
> ├── main.py
├── database.py
├── models.py
├── schemas.py
├── repository.py
├── static/
│   ├── index.html
│   ├── attendance.html
│   └── proofs/
├── requirements.txt
├── .env


## Database Schema
### Students
	•	id
	•	name
	•	enrollment_number

### Student Embeddings
	•	id
	•	student_id (FK)
	•	embedding (vector)

### Attendance Session
	•	id
	•	subject_name
	•	timestamp
	•	proof_image_path

### Attendance Log
	•	id
	•	session_id (FK)
	•	student_id (FK)
	•	is_present
	•	confidence_score

 ## Setup Instructions

1️⃣ Clone Repository

git clone https://github.com/face-attendance-system/face-recognition-attendance.git
cd face-recognition-attendance


2️⃣ Create Virtual Environment

python3 -m venv venv
source venv/bin/activate


3️⃣ Install Dependencies

pip install -r requirements.txt


4️⃣ Setup PostgreSQL

CREATE DATABASE attendance_db;
CREATE EXTENSION vector;


5️⃣ Configure Environment

Create .env file:

DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/attendance_db


6️⃣ Run Server

uvicorn main:app --reload


🌐 Frontend Usage

🔹 Registration Page (Stage 2)

http://127.0.0.1:8000/

🔹 Attendance System (Stage 3)

http://127.0.0.1:8000/attendance


🔌 API Endpoints

📌 Register Student (Video)

POST /register-video/


📌 Analyze Group Photo

POST /analyze-group-photo/

Returns:
	•	Detected faces
	•	Matched students
	•	Confidence scores
	•	Annotated image

⸻

📌 Submit Attendance

POST /submit-attendance/

Stores final verified attendance

⸻

⚡ Challenges Solved

🔹 Multiple Face Matching

Used Hungarian Algorithm to avoid duplicate assignments

🔹 Low Quality Faces
	•	Upscaling images
	•	CLAHE enhancement
	•	Blur filtering

🔹 Embedding Stability

Used averaged embeddings for better accuracy

⸻

## Future Improvements
	•	🎥 Live webcam attendance
	•	📊 Analytics dashboard
	•	🔐 Authentication (JWT)
	•	☁️ Cloud deployment
	•	⚡ GPU acceleration

