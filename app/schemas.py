from typing import List
from pydantic import BaseModel


class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: Study
    prior_studies: List[Study]


class RequestSchema(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str
    cases: List[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool
    confidence: float


class ResponseSchema(BaseModel):
    predictions: List[Prediction]
