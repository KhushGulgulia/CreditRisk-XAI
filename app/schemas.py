from pydantic import BaseModel, Field
from typing import List

class CreditProfile(BaseModel):
    LIMIT_BAL: float = Field(..., ge=1000, le=1000000, example=200000)
    SEX: int = Field(..., ge=1, le=2, example=1)
    EDUCATION: int = Field(..., ge=1, le=4, example=1)
    MARRIAGE: int = Field(..., ge=1, le=3, example=2)
    AGE: int = Field(..., ge=18, le=100, example=32)
    PAY_0: int = Field(..., ge=-2, le=8, example=-1)
    PAY_2: int = Field(..., ge=-2, le=8, example=-1)
    PAY_3: int = Field(..., ge=-2, le=8, example=-1)
    PAY_4: int = Field(..., ge=-2, le=8, example=-1)
    PAY_5: int = Field(..., ge=-2, le=8, example=-1)
    PAY_6: int = Field(..., ge=-2, le=8, example=-1)
    BILL_AMT1: float = Field(..., example=5000)
    BILL_AMT2: float = Field(..., example=4000)
    BILL_AMT3: float = Field(..., example=3000)
    BILL_AMT4: float = Field(..., example=2000)
    BILL_AMT5: float = Field(..., example=1000)
    BILL_AMT6: float = Field(..., example=500)
    PAY_AMT1: float = Field(..., ge=0, example=5000)
    PAY_AMT2: float = Field(..., ge=0, example=4000)
    PAY_AMT3: float = Field(..., ge=0, example=3000)
    PAY_AMT4: float = Field(..., ge=0, example=2000)
    PAY_AMT5: float = Field(..., ge=0, example=1000)
    PAY_AMT6: float = Field(..., ge=0, example=500)

class AdverseFactor(BaseModel):
    feature: str
    impact_score: float
    direction: str

class RiskAssessmentResponse(BaseModel):
    risk_level: str
    default_probability: float
    decision_threshold: float
    top_adverse_factors: List[AdverseFactor]
    latency_ms: float