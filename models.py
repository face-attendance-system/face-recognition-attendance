# models.py
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    enrollment_number = Column(String, unique=True, index=True)
    
    # Establish a relationship to the embeddings table
    embeddings = relationship("StudentEmbedding", back_populates="student", cascade="all, delete-orphan")
    
    attendance_records = relationship("AttendanceLog", back_populates="student", cascade="all, delete-orphan")

class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), index=True)
    
    # 512-dimensional vector for ArcFace
    embedding = Column(Vector(512)) 
    
    student = relationship("Student", back_populates="embeddings")

# --- NEW STAGE 3 MODELS ---

class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    subject_name = Column(String, index=True)
    date = Column(DateTime(timezone=True), server_default=func.now())
    proof_image_path = Column(String) # Path to the saved image with bounding boxes
    
    logs = relationship("AttendanceLog", back_populates="session", cascade="all, delete-orphan")

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id", ondelete="CASCADE"), index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), index=True)
    
    is_present = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True) # Matches the confidence percentage from your sketch
    
    session = relationship("AttendanceSession", back_populates="logs")
    student = relationship("Student", back_populates="attendance_records")