# app/backend/app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .models import ChatRequest, ChatResponse
from .services.chat_service import chat_service_instance

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Attempting to load model on startup...")
    # The model is loaded when chat_service_instance is created (singleton pattern).
    # We just check its status here.
    if chat_service_instance._model is None or chat_service_instance._tokenizer is None:
        print("WARNING: Model could not be loaded on startup. Check chat_service.py logs for details.")
        # Depending on requirements, might want to prevent the app from starting
        # if the model is critical and fails to load.
        # For now, it will start with a warning, and the /chat endpoint will return 503.
    else:
        print("Chatbot model loaded and ready.")
    yield
    # Code to run on shutdown (if any)
    print("Application shutting down...")

app = FastAPI(
    title="SophiaWeaver Chatbot API", # UPDATED
    description="API for interacting with the GenAI Chatbot focused on The Bible.", # UPDATED
    version="0.1.0",
    lifespan=lifespan
)



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