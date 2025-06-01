# app/backend/app/services/chat_service.py
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class ChatService:
    _instance = None
    _model: GPT2LMHeadModel | None = None  # Added type hint
    _tokenizer: GPT2Tokenizer | None = None # Added type hint
    _device: torch.device | None = None # Added type hint for device

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
            # Model loading is initiated here, but success isn't guaranteed
            # The actual loading happens in _load_model, called by the constructor
            try:
                cls._instance._load_model()
            except RuntimeError as e:
                # If _load_model raises a RuntimeError (e.g., can't load any model),
                # _model and _tokenizer will remain None.
                # The lifespan event in main.py will catch this via is_model_ready().
                print(f"ChatService: Critical error during initial model load: {e}")
        return cls._instance

        # app/backend/app/services/chat_service.py
        # ... (other parts of the class) ...

    def _load_model(self):
        if ChatService._model is None or ChatService._tokenizer is None:
            # Correctly navigate to the project root "SophiaWeaver/"
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parent.parent.parent.parent.parent

            model_path = project_root / "trained_models" / "the_bible_gpt2_small"

            print(f"DEBUG: Current file __file__: {current_file_path}")
            print(f"DEBUG: Calculated project_root: {project_root}")
            print(f"DEBUG: Attempting to load model from absolute path: {model_path.resolve()}")

            if not model_path.exists() or not model_path.is_dir():
                print(f"Model directory not found or is not a directory: {model_path.resolve()}")
                print("Please ensure the model is trained and available in the correct location.")
                # Optional: List contents of expected parent to help debug
                try:
                    print(f"Listing contents of {project_root / 'trained_models'}:")
                    print(list((project_root / "trained_models").iterdir()))
                except FileNotFoundError:
                    print(f"Directory not found: {project_root / 'trained_models'}")

                print("Attempting to load base 'gpt2' model as a fallback.")
                model_name_or_path = "gpt2"
            else:
                print(f"Model directory found: {model_path.resolve()}")
                # Optional: List contents of model directory
                print("Listing contents of model directory:")
                print(list(model_path.iterdir()))
                model_name_or_path = str(model_path)

            try:
                print(f"Attempting to load tokenizer and model from: {model_name_or_path}")
                ChatService._tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
                ChatService._model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

                if ChatService._tokenizer.pad_token is None:
                    ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token

                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use self._device
                ChatService._model.to(self._device)
                ChatService._model.eval()
                print(f"Model loaded successfully from '{model_name_or_path}' on {self._device}.")
            except Exception as e:
                print(f"CRITICAL: Failed to load model from '{model_name_or_path}': {e}")
                if model_name_or_path != "gpt2":
                    print("Attempting to load base 'gpt2' model as a fallback due to previous error.")
                    try:
                        ChatService._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                        ChatService._model = GPT2LMHeadModel.from_pretrained("gpt2")

                        if ChatService._tokenizer.pad_token is None:
                            ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token

                        self._device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu")  # Use self._device
                        ChatService._model.to(self._device)
                        ChatService._model.eval()
                        print(f"Base 'gpt2' model loaded successfully on {self._device}.")
                    except Exception as fallback_e:
                        print(f"CRITICAL: Failed to load base 'gpt2' model as fallback: {fallback_e}")
                        ChatService._model = None  # Ensure they are None
                        ChatService._tokenizer = None
                        self._device = None
                        raise RuntimeError(
                            f"Could not load any model. Primary error: {e}. Fallback error: {fallback_e}") from fallback_e
                else:
                    ChatService._model = None  # Ensure they are None
                    ChatService._tokenizer = None
                    self._device = None
                    raise RuntimeError(f"Could not load base 'gpt2' model: {e}") from e


    def is_model_ready(self) -> bool:
        """Checks if the model and tokenizer are loaded and ready."""
        return ChatService._model is not None and ChatService._tokenizer is not None

    def generate_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        if not self.is_model_ready():  # Use the public method
            # This case should ideally be caught by the /chat endpoint's check first
            return "Error: Model not available. Please check server logs."

        # Ensure _model and _tokenizer are not None here, which is guaranteed by is_model_ready()
        # but linters/type checkers might still want explicit checks or asserts
        # if they can't infer it perfectly. For runtime, is_model_ready() is the guard.
        assert ChatService._model is not None
        assert ChatService._tokenizer is not None
        assert self._device is not None

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
            print(f"Error during generation in ChatService: {e}") # Log here too
            # Re-raise or return a generic error message.
            # For the API, it's often better to let the endpoint handle the HTTP exception.
            raise # Or return "Sorry, an error occurred." and let the endpoint wrap it


chat_service_instance = ChatService()