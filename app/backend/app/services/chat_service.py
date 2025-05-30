# app/backend/app/services/chat_service.py
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class ChatService:
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if ChatService._model is None or ChatService._tokenizer is None:
            project_root = Path(__file__).parent.parent.parent  # SophiaWeaver/
            print(f"Project root: {project_root}")

            model_path = project_root / "trained_models" / "the_bible_gpt2_small"  # UPDATED

            print(f"Loading model from: {model_path}")
            if not model_path.exists():
                print(f"Model directory not found: {model_path}")
                print("Please ensure the model is trained and available.")
                print("Attempting to load base 'gpt2' model as a fallback.")
                model_name_or_path = "gpt2"
            else:
                model_name_or_path = str(model_path)

            try:
                print(f"Attempting to load from: {model_name_or_path}")
                ChatService._tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
                ChatService._model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

                if ChatService._tokenizer.pad_token is None:
                    ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ChatService._model.to(self.device)
                ChatService._model.eval()  # Set model to evaluation mode
                print(f"Model loaded successfully on {self.device}.")
            except Exception as e:
                print(f"CRITICAL: Failed to load model from explicitly set path '{model_name_or_path}': {e}")
                # raise RuntimeError(f"Forced stop: Could not load model from {model_name_or_path}") # Force stop
                # OR just let it be None and see the warning in main.py
                ChatService._model = None
                ChatService._tokenizer = None
                raise RuntimeError(f"Could not load model: {e}")

    def generate_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        if self._model is None or self._tokenizer is None:
            return "Error: Model not loaded. Please check server logs."

        try:
            prompt = user_input
            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            outputs = self._model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95
            )

            response_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response_text.startswith(prompt):
                bot_reply = response_text[len(prompt):].strip()
            else:
                bot_reply = response_text.strip()

            if not bot_reply:
                bot_reply = "I'm not sure how to respond to that. Could you try rephrasing?"

            return bot_reply

        except Exception as e:
            print(f"Error during generation: {e}")
            return "Sorry, I encountered an error while generating a response."


chat_service_instance = ChatService()