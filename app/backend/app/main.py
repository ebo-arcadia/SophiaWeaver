# app/backend/app/main.py
from fastapi import FastAPI, HTTPException
from .models import ChatRequest, ChatResponse
from .services.chat_service import chat_service_instance

app = FastAPI(
    title="SophiaWeaver Chatbot API", # UPDATED
    description="API for interacting with the GenAI Chatbot focused on The Bible.", # UPDATED
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    if chat_service_instance._model is None:
        print("WARNING: Model could not be loaded on startup. Check chat_service.py logs.")
    else:
        print("Chatbot model loaded and ready.")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if chat_service_instance._model is None or chat_service_instance._tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")
    try:
        bot_response_text = chat_service_instance.generate_response(
            user_input=request.user_input,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return ChatResponse(bot_response=bot_response_text)
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": f"Welcome to the SophiaWeaver Chatbot API. Use the /chat endpoint to interact."} # UPDATED