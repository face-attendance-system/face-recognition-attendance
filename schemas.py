from pydantic import BaseModel, field_validator
from typing import List

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