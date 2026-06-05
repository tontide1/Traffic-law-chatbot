from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    stream: Optional[bool] = False
    comparison_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    mode: str = "hybrid"
    sources: Optional[List[Dict[str, Any]]] = []

class ComparisonResponse(BaseModel):
    naive: ChatResponse
    hybrid: ChatResponse

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
