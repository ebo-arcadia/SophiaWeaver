# app/frontend/app.py
import streamlit as st
import requests
import os  # Import the os module
from requests.models import Response # Import Response for type hinting

# Get the backend API URL from the environment variable
# Provide a default for local development outside of Docker
BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

st.title("SophiaWeaver - Bible Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something about The Bible..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = ""
        try:
            # Prepare the request payload
            payload = {
                "user_input": prompt,
                "max_length": 150,  # You can adjust this
                "temperature": 0.7  # You can adjust this
            }
            # Send request to the backend
            response: Response = requests.post(CHAT_ENDPOINT, json=payload)  # Use the CHAT_ENDPOINT
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            bot_response_data = response.json()
            full_response_text = bot_response_data.get("bot_response", "Sorry, I couldn't get a response.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the backend: {e}")
            full_response_text = "Error: Could not connect to the chatbot service."
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            full_response_text = "Error: An unexpected error occurred while fetching the response."

        message_placeholder.markdown(full_response_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response_text})