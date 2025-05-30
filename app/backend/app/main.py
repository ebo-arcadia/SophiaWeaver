# app/backend/app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .models import ChatRequest, ChatResponse
from .services.chat_service import chat_service_instance

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan: Startup sequence initiated.")
    # The chat_service_instance is already created when imported.
    # We use its public method to check readiness.
    if not chat_service_instance.is_model_ready(): # Use the public method
        print("WARNING (Lifespan): Model is not ready on startup. Check chat_service.py logs.")
    else:
        print("Lifespan: Chatbot model is ready.")
    yield
    print("Lifespan: Shutdown sequence initiated.")

app = FastAPI(
    title="SophiaWeaver Chatbot API", # UPDATED
    description="API for interacting with the GenAI Chatbot focused on The Bible.", # UPDATED
    version="0.1.0",
    lifespan=lifespan
)



@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not chat_service_instance.is_model_ready():
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