from pydantic import BaseModel
from typing import List

class StudentCreate(BaseModel):
    name: str
    enrollment_number: str
    face_embedding: List[float]

class ProbeRequest(BaseModel):          # ← Add this
    face_embedding: List[float]

class StudentResponse(BaseModel):
    id: int
    name: str
    enrollment_number: str

    class Config:
        from_attributes = True