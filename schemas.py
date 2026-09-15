from pydantic import BaseModel, Field
from typing import Literal

class IntentResult(BaseModel):
    intent: Literal[
        "DELIVERY_DELAY",
        "REFUND_AND_RETURN", 
        "DIGITAL_CONTENT_APP",
        "ORDER_STATUS",
        "WRONG_DAMAGED_ITEM",
        "ACCOUNT_ACCESS",
        "UNCLEAR_VAGUE"
    ]
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0")
    reasoning: str = Field(description="Brief reason for choosing this intent")