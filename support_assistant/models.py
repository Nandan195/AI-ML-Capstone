from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
