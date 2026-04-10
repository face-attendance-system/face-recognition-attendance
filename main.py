from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import engine, Base, get_db
import models, schemas
from repository import StudentRepository
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Face Recognition Attendance Backend")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register/", response_model=schemas.StudentResponse)
def register_student(student: schemas.StudentCreate, db: Session = Depends(get_db)):
    return StudentRepository.create_student(db, student)

@app.post("/recognize/")
def recognize_face(probe: schemas.ProbeRequest, db: Session = Depends(get_db)):  # Bug 1 fixed
    matches = StudentRepository.find_matching_student(db, probe.face_embedding)
    if matches:
        return {"match_found": True, "student_name": matches[0].name}  # Bug 2 fixed
    return {"match_found": False}