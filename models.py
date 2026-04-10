from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from database import Base

class StudentModel(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    enrollment_number = Column(String, unique=True, index=True)
    face_embedding = Column(Vector(512))