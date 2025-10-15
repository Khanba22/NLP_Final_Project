
# app/api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class SummarizationResponse(BaseModel):
    summary: str
    method: str
    original_length: int
    summary_length: int

class ComparisonResponse(BaseModel):
    summaries: List[Dict[str, Any]]
    common_themes: List[str]
    unique_points: Dict[str, List[str]]
    method: str