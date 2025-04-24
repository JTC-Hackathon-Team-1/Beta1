# app/schemas/pipeline.py

from pydantic import BaseModel
from typing import List, Dict

class Entity(BaseModel):
    text: str
    label: str

class PipelineRequest(BaseModel):
    session_id: str
    source_language: str    # e.g. "en", "es", "legalese"
    target_language: str    # ignored: we always output English
    text: str

class PipelineResponse(BaseModel):
    translation: str        # final English text (possibly simplified)
    entities: List[Entity]
    veracity_score: float
    bias_score: float
    audit_log: Dict[str, float]