from sqlalchemy.orm import Session
from models import StudentModel
from schemas import StudentCreate

class StudentRepository:
    @staticmethod
    def create_student(db: Session, student: StudentCreate):
        db_student = StudentModel(
            name=student.name,
            enrollment_number=student.enrollment_number,
            face_embedding=student.face_embedding
        )
        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        return db_student

    @staticmethod
    def find_matching_student(db: Session, probe_embedding: list[float], limit: int = 1):
        results = db.query(StudentModel).order_by(
            StudentModel.face_embedding.cosine_distance(probe_embedding)
        ).limit(limit).all()
        return results