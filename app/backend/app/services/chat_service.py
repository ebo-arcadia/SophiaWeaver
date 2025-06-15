# app/backend/app/services/chat_service.py
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import os


class ChatService:
    # Class attributes to hold the model, tokenizer, device, and ready state
    _model: GPT2LMHeadModel | None = None
    _tokenizer: GPT2Tokenizer | None = None
    _device: torch.device | None = None
    _model_ready: bool = False  # Flag to indicate if model loading has been attempted and its status

    MIN_INPUT_LENGTH = 3
    CONTAINS_ALPHANUMERIC_PATTERN = re.compile(r'[a-zA-Z0-9]')

    def __init__(self):
        """
        Initializes the ChatService.
        The model loading is triggered here if it hasn't been loaded yet.
        """
        if not ChatService._model_ready:  # Check the class-level flag
            # Call _load_model which will attempt to load and set class attributes
            self._load_model()

    def _load_model(self):
        """
        Loads the model and tokenizer and sets them as class attributes.
        This method is called only if the model hasn't been loaded yet.
        Sets the _model_ready flag based on the outcome.
        """
        # This check ensures that even if _load_model is somehow called again,
        # it won't reload if already successful. The primary guard is in __init__.
        if ChatService._model is not None and ChatService._tokenizer is not None:
            print("Model and tokenizer are already loaded (checked in _load_model).")
            ChatService._model_ready = True  # Ensure flag is consistent
            return

        print("Attempting to load model and tokenizer for the first time...")
        container_workdir = Path("/app/backend")
        model_base_path = container_workdir
        project_root_for_models = model_base_path
        model_path = project_root_for_models / "trained_models" / "the_bible_gpt2_small"

        print(f"DEBUG: chat_service.py _load_model - Effective model base path: {project_root_for_models}")
        print(f"DEBUG: chat_service.py _load_model - Attempting to load model from: {model_path.resolve()}")

        model_name_or_path_to_load: str
        fine_tuned_model_exists = model_path.exists() and model_path.is_dir()

        if fine_tuned_model_exists:
            print(f"Fine-tuned model directory found: {model_path.resolve()}")
            model_name_or_path_to_load = str(model_path)
        else:
            expected_parent = project_root_for_models / "trained_models"
            print(f"Fine-tuned model directory not found or is not a directory: {model_path.resolve()}")
            print("Please ensure the model is trained and available in the correct location.")
            try:
                print(f"Listing contents of {expected_parent}:")
                if expected_parent.exists():
                    print(list(expected_parent.iterdir()))
                else:
                    print(f"Directory {expected_parent} does not exist.")
            except Exception as list_e:
                print(f"Error listing directory {expected_parent}: {list_e}")
            print("Attempting to load base 'gpt2' model as a fallback.")
            model_name_or_path_to_load = "gpt2"

        try:
            print(f"Attempting to load tokenizer and model from: {model_name_or_path_to_load}")
            ChatService._tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path_to_load)
            ChatService._model = GPT2LMHeadModel.from_pretrained(model_name_or_path_to_load)

            if ChatService._tokenizer.pad_token is None:
                ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token

            ChatService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ChatService._model.to(ChatService._device)
            ChatService._model.eval()
            print(f"Model loaded successfully from '{model_name_or_path_to_load}' on {ChatService._device}.")
            ChatService._model_ready = True
        except Exception as e:
            print(f"CRITICAL: Failed to load model from '{model_name_or_path_to_load}': {e}")
            ChatService._model = None
            ChatService._tokenizer = None
            ChatService._model_ready = False  # Explicitly set to false on any failure

            # If loading the fine-tuned model failed, and it wasn't already trying to load 'gpt2'
            if fine_tuned_model_exists and model_name_or_path_to_load != "gpt2":
                print("Attempting to load base 'gpt2' model as a final fallback due to previous error.")
                try:
                    ChatService._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    ChatService._model = GPT2LMHeadModel.from_pretrained("gpt2")
                    if ChatService._tokenizer.pad_token is None:
                        ChatService._tokenizer.pad_token = ChatService._tokenizer.eos_token
                    ChatService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ChatService._model.to(ChatService._device)
                    ChatService._model.eval()
                    print(f"Base 'gpt2' model loaded successfully on {ChatService._device}.")
                    ChatService._model_ready = True  # Success with fallback
                except Exception as fallback_e:
                    print(f"CRITICAL: Failed to load base 'gpt2' model as fallback: {fallback_e}")
                    ChatService._model = None
                    ChatService._tokenizer = None
                    ChatService._model_ready = False  # All attempts failed
            # If even 'gpt2' failed initially, _model_ready is already False

    def is_model_ready(self) -> bool:
        """Checks if the model and tokenizer are loaded and ready."""
        # _model_ready flag is the primary indicator.
        # Additionally, ensure model and tokenizer objects are not None.
        return ChatService._model_ready and ChatService._model is not None and ChatService._tokenizer is not None

    def generate_response(self, user_input: str, max_length: int = 100, temperature: float = 0.7) -> str:
        if not self.is_model_ready():
            return "Error: Model not available. Please check server logs."

        sanitized_input = user_input.strip()

        if len(sanitized_input) < self.MIN_INPUT_LENGTH:
            return "I need a bit more to go on! Could you please provide a longer question or statement? 🤔"

        if not self.CONTAINS_ALPHANUMERIC_PATTERN.search(sanitized_input):
            return "I'm having trouble understanding that. Could you try rephrasing with more standard text, perhaps including some letters or numbers? 🧐"

        # At this point, is_model_ready() was true, so class attributes should be set.
        # Using asserts for linters/type checkers and as a strong guarantee.
        assert ChatService._model is not None
        assert ChatService._tokenizer is not None
        assert ChatService._device is not None

        try:
            prompt = sanitized_input
            tokenized_inputs = ChatService._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = tokenized_inputs['input_ids'].to(ChatService._device)
            attention_mask = tokenized_inputs['attention_mask'].to(ChatService._device)

            input_sequence_length = inputs.shape[1]
            outputs = ChatService._model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=input_sequence_length + max_length,
                num_return_sequences=1,
                pad_token_id=ChatService._tokenizer.pad_token_id,
                eos_token_id=ChatService._tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95
            )

            response_text = ChatService._tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response_text.startswith(prompt):
                bot_reply = response_text[len(prompt):].strip()
            else:
                bot_reply = response_text.strip()

            if not bot_reply:
                bot_reply = "I'm not sure how to respond to that. Could you try rephrasing?"

            return bot_reply

        except Exception as e:
            print(f"Error during generation in ChatService: {e}")
            # Depending on your error handling strategy for the API,
            # you might want to raise a custom exception or return a generic error.
            # For now, returning a string.
            return "Sorry, an error occurred while generating a response."


# This line will create an instance of ChatService when the module is imported.
# The __init__ method will be called, triggering _load_model if needed.
# This effectively loads the model at application startup (when this module is first imported).
chat_service_instance = ChatService()
