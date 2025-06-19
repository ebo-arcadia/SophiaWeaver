# tests/backend/test_chat_service.py

import pytest
from unittest.mock import patch, MagicMock

# Import the ChatService class directly
# Note: Importing the module might trigger the model loading due to the
# instance creation at the bottom of chat_service.py.
# For proper unit testing, you'd typically mock the model loading process.
# We'll address that briefly below, but for testing validation,
# we can often get away with just importing the class.
from app.backend.app.services.chat_service import ChatService


# --- Mocking the Model Loading ---
# To prevent the actual model from loading during tests (which is slow and
# requires the model files), we can mock the _load_model method.
# This patch needs to be applied before any test that might instantiate ChatService.
# Using autouse=True on a fixture is a common pytest pattern for this.

@pytest.fixture(autouse=True)
def mock_model_loading():
    """Fixture to mock the _load_model method for all tests."""
    with patch('app.backend.app.services.chat_service.ChatService._load_model') as mock_load:
        # We also need to simulate that the model is "ready" after loading
        # for the is_model_ready check to pass for valid inputs.
        # In a real scenario, you might mock the model and tokenizer objects too.
        ChatService._model_ready = True
        ChatService._model = MagicMock()  # Simulate a loaded model object
        ChatService._tokenizer = MagicMock()  # Simulate a loaded tokenizer object
        ChatService._device = MagicMock()  # Simulate a device object

        # Configure the mocked tokenizer's decode method if needed for output tests
        # mock_tokenizer.decode.return_value = "Mocked bot response"

        yield mock_load  # The test runs here

    # --- Cleanup after each test ---
    # Reset class attributes to their initial state
    ChatService._model_ready = False
    ChatService._model = None
    ChatService._tokenizer = None
    ChatService._device = None


# --- Tests for Input Validation ---

# Use parametrize to test multiple invalid short inputs
@pytest.mark.parametrize("short_input", ["a", "hi", "", "   ", "ab"])
def test_generate_response_short_input(short_input):
    """Tests that short inputs return the specific short input error message."""
    chat_service = ChatService()  # Instantiate the service (model loading is mocked)
    expected_error_message = "I need a bit more to go on! Could you please provide a longer question or statement? 🤔"
    response = chat_service.generate_response(short_input)
    assert response == expected_error_message


# Use parametrize to test multiple inputs with only special characters
@pytest.mark.parametrize("special_char_input", ["!!!", "???", "#$%", "!@#$%", "  !@#  "])
def test_generate_response_special_chars_input(special_char_input):
    """Tests that inputs with only special characters return the specific special chars error message."""
    chat_service = ChatService()  # Instantiate the service (model loading is mocked)
    expected_error_message = "I'm having trouble understanding that. Could you try rephrasing with more standard text, perhaps including some letters or numbers? 🧐"
    response = chat_service.generate_response(special_char_input)
    assert response == expected_error_message


# Test a valid input to ensure validation *passes*
def test_generate_response_valid_input():
    """Tests that a valid input does NOT return a validation error message."""
    chat_service = ChatService()  # Instantiate the service (model loading is mocked)
    valid_input = "What is the meaning of faith?"

    # Call the method
    response = chat_service.generate_response(valid_input)

    # Assert that the response is NOT one of the validation error messages
    # This implies the validation checks were passed.
    # In a more complete test, you would also assert that the mocked model
    # generation methods were called with the correct arguments.
    assert response != "I need a bit more to go on! Could you please provide a longer question or statement? 🤔"
    assert response != "I'm having trouble understanding that. Could you try rephrasing with more standard text, perhaps including some letters or numbers? 🧐"
    # You might also assert that the response is not empty or is a mocked model output
    # assert response == "Mocked bot response" # If you configured the mock decode return value
    # Or simply check it's not empty if the mock returns something non-empty by default
    assert response is not None  # Or a more specific check based on mock behavior

# --- Example of how you might test the fallback if model is NOT ready ---
# You'd need to run this test without the autouse=True mock fixture above,
# or create a specific fixture that *doesn't* mock _load_model or sets _model_ready=False

# def test_generate_response_model_not_ready():
#     """Tests the response when the model is not ready."""
#     # Ensure model is not ready for this specific test
#     ChatService._model_ready = False
#     ChatService._model = None
#     ChatService._tokenizer = None
#     ChatService._device = None

#     chat_service = ChatService()
#     user_input = "Any question"
#     expected_error = "Error: Model not available. Please check server logs."
#     response = chat_service.generate_response(user_input)
#     assert response == expected_error
