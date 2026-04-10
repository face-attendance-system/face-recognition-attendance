# Face Recognition Attendance System

A smart attendance system using Face Recognition with FastAPI, PostgreSQL, and pgvector.

This project is built in stages:
- Stage 1: Backend + Database setup
- Stage 2: Face Recognition integration (coming / in progress)

## Tech Stack

- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- OpenCV (Stage 2)

## Project Structure

.
├── main.py
├── database.py
├── models.py
├── schemas.py
├── repository.py
├── static/
│   └── index.html
├── requirements.txt

## Project Stages

### Stage 1: Backend + Database
- FastAPI server setup
- PostgreSQL connection
- pgvector integration
- Student registration API
- Face embedding storage

## Setup Instructions

1. Clone the repo:
git clone https://github.com/face-attendance-system/face-recognition-attendance.git

2. Create virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Setup PostgreSQL:
- Create database: attendance_db
- Enable extension:
  CREATE EXTENSION vector;

5. Add .env file:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/attendance_db

6. Run server:
uvicorn main:app --reload


## API Endpoints

POST /register/ → Register student  
POST /recognize/ → Recognize face  
GET / → Frontend


## Future Improvements

- Add frontend UI dashboard
- Improve recognition accuracy
- Add live camera support
- Deploy using Docker
