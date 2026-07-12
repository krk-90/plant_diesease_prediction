from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DiseaseClass(str, Enum):
    CANKER = "Canker"
    GREENING = "Greening"
    GUMMOSIS = "Gummosis"
    HEALTHY = "Healthy"
    LEAF_MINER = "Leaf-miner"
    LEMON_BUTTERFLY = "Lemon-butterfly"


class PredictionResult(BaseModel):
    prediction: DiseaseClass
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

class PredictionResponse(PredictionResult):
    filename: str
    request_id: Optional[str] = None
    inference_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int = 6
    
class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
