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

    def _load_model(self):
        if ChatService._model is None or ChatService._tokenizer is None:
            # Determine the base path for models.
            # When running in Docker, the WORKDIR is /app/backend.
            # The volume mount places 'trained_models' inside this WORKDIR.
            # So, the effective project_root *inside the container* for model loading
            # should be the WORKDIR itself.

            # A more robust way to define where models are, especially in Docker:
            # Option 1: Assume models are relative to the service's working directory
            # This aligns with how Docker volumes are often set up.
            # The WORKDIR in your backend Dockerfile is /app/backend
            container_workdir = Path("/app/backend")
            model_base_path = container_workdir

            # Option 2: Use an environment variable to specify the models directory (more flexible)
            # For example, you could set MODELS_DIR_PATH=/app/backend/trained_models in docker-compose.yml
            # models_dir_env = os.getenv("MODELS_DIR_PATH")
            # if models_dir_env:
            #     model_base_path = Path(models_dir_env).parent # if MODELS_DIR_PATH points directly to trained_models
            # else:
            #     # Fallback for local development or if env var not set
            #     current_file_path = Path(__file__).resolve()
            #     # For local: SoftwareDev/GenAI/SophiaWeaver/app/backend/app/services/chat_service.py
            #     # project_root should be SoftwareDev/GenAI/SophiaWeaver
            #     model_base_path = current_file_path.parent.parent.parent.parent.parent

            project_root_for_models = model_base_path

            model_path = project_root_for_models / "trained_models" / "the_bible_gpt2_small"

            print(f"DEBUG: chat_service.py _load_model - Effective model base path: {project_root_for_models}")
            print(f"DEBUG: chat_service.py _load_model - Attempting to load model from: {model_path.resolve()}")

            if not model_path.exists() or not model_path.is_dir():
                print(f"Model directory not found or is not a directory: {model_path.resolve()}")
                print("Please ensure the model is trained and available in the correct location.")
                # Optional: List contents of expected parent to help debug
                try:
                    expected_parent = project_root_for_models / "trained_models"
                    print(f"Listing contents of {expected_parent}:")
                    if expected_parent.exists():
                        print(list(expected_parent.iterdir()))
                    else:
                        print(f"Directory {expected_parent} does not exist.")
                except Exception as list_e:
                    print(f"Error listing directory {expected_parent}: {list_e}")

                print("Attempting to load base 'gpt2' model as a fallback.")
                model_name_or_path = "gpt2"
            else:
                print(f"Model directory found: {model_path.resolve()}")
                model_name_or_path = str(model_path)

            try:
                print(f"Attempting to load tokenizer and model from: {model_name_or_path}")
                ChatService._tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
                ChatService._model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

                if ChatService._tokenizer.pad_token is None:
                    ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token

                ChatService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ChatService._model.to(ChatService._device)
                ChatService._model.eval()
                print(f"Model loaded successfully from '{model_name_or_path}' on {ChatService._device}.")
                ChatService._model_ready = True
            except Exception as e:
                print(f"CRITICAL: Failed to load model from '{model_name_or_path}': {e}")
                # Fallback logic (simplified for brevity, ensure your full fallback is robust)
                if model_name_or_path != "gpt2":  # Avoid infinite loop if base gpt2 fails
                    print("Attempting to load base 'gpt2' model as a fallback due to previous error.")
                    try:
                        ChatService._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                        ChatService._model = GPT2LMHeadModel.from_pretrained("gpt2")
                        if ChatService._tokenizer.pad_token is None:
                            ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token
                        ChatService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        ChatService._model.to(ChatService._device)
                        ChatService._model.eval()
                        print(f"Base 'gpt2' model loaded successfully on {ChatService._device}.")
                        ChatService._model_ready = True
                    except Exception as fallback_e:
                        print(f"CRITICAL: Failed to load base 'gpt2' model as fallback: {fallback_e}")
                        # Ensure model and tokenizer are None if all attempts fail
                        ChatService._model = None
                        ChatService._tokenizer = None
                        ChatService._model_ready = False
                        # Consider re-raising or handling this critical failure appropriately
                else:  # Base gpt2 itself failed
                    ChatService._model = None
                    ChatService._tokenizer = None
                    ChatService._model_ready = False
        else:
            print("Model and tokenizer already loaded.")


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

            # Use the tokenizer's __call__ method for robust tensor output
            # This returns a BatchEncoding object (like a dictionary)
            tokenized_inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,  # Optional: consider padding strategy if batching
                truncation=True, # Optional: consider truncation if inputs can be too long
                max_length=512
            )

            inputs = tokenized_inputs['input_ids'].to(self._device)
            attention_mask = tokenized_inputs['attention_mask'].to(self._device)

            input_sequence_length = inputs.shape[1]
            outputs = self._model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=input_sequence_length + max_length,
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