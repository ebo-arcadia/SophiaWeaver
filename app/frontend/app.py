# app/frontend/app.py
import streamlit as st
import requests
import os

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="SophiaWeaver Chat", layout="wide")  # UPDATED

st.title("💬 SophiaWeaver Chatbot")  # UPDATED
st.caption("A GenAI chatbot for insightful conversations, focused on The Bible.")  # UPDATED

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hello! How can I help you explore insights from The Bible today?"}]  # Initial message

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about The Bible..."):  # UPDATED placeholder
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            payload = {
                "user_input": prompt,
                "max_length": 150,
                "temperature": 0.7
            }
            response = requests.post(BACKEND_API_URL, json=payload, timeout=120)
            response.raise_for_status()

            bot_response_data = response.json()
            full_response = bot_response_data.get("bot_response", "Sorry, I didn't get a valid response.")

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to backend: {e}"
            st.error(full_response)
        except Exception as e:
            full_response = f"An unexpected error occurred: {e}"
            st.error(full_response)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("About SophiaWeaver")  # UPDATED
    st.markdown("This is Phase 1 of the **SophiaWeaver** chatbot, fine-tuned on texts from The Bible.")  # UPDATED
    st.markdown("---")
    st.subheader("Backend API URL")
    st.text_input("API URL", value=BACKEND_API_URL, key="api_url_display", disabled=True)
    st.caption(
        "This is the URL the frontend uses to talk to the backend. Configurable via BACKEND_API_URL environment variable.")