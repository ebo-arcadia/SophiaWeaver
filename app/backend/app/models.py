# app/backend/app/models.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input: str
    max_length: int = 100 # Max tokens for the generated response
    temperature: float = 0.7 # Controls randomness: lower is more deterministic

class ChatResponse(BaseModel):
    bot_response: str