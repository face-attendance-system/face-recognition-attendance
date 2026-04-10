# models.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    enrollment_number = Column(String, unique=True, index=True)
    
    # Establish a relationship to the embeddings table
    embeddings = relationship("StudentEmbedding", back_populates="student", cascade="all, delete-orphan")

class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), index=True)
    
    # 512-dimensional vector for ArcFace
    embedding = Column(Vector(512)) 
    
    student = relationship("Student", back_populates="embeddings")