from pydantic import BaseModel, field_validator
from typing import List, Optional

class StudentRegisterBase(BaseModel):
    name: str
    enrollment_number: str
    face_embeddings: List[List[float]]

    @field_validator("face_embeddings")
    @classmethod
    def check_embedding_dimensions(cls, embeddings):
        for i, emb in enumerate(embeddings):
            if len(emb) != 512:
                raise ValueError(f"Embedding #{i+1} has {len(emb)} dims, expected 512")
        return embeddings
    
# --- NEW STAGE 3 SCHEMAS ---

# Represents a single student's tick box on the frontend
class StudentAttendanceUpdate(BaseModel):
    student_id: int
    is_present: bool
    confidence_score: Optional[float] = None

# Represents the final payload sent when the teacher clicks "Mark Attendance"
class AttendanceSubmit(BaseModel):
    subject_name: str
    proof_image_url: str  # We receive the reviewed image to save to disk
    records: List[StudentAttendanceUpdate]