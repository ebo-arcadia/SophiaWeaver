# app/frontend/app.py
import streamlit as st
import requests
import os
import time # For a slight delay to show "thinking"
from requests.models import Response

# --- Page Configuration (Set this at the very top) ---
st.set_page_config(
    page_title="SophiaWeaver ✨",
    page_icon="📖",  # Bible emoji or a custom icon
    layout="centered",   # Changed from "wide", "centered" works well with a sidebar
    initial_sidebar_state="expanded" # Ensure sidebar is visible
)

# --- Backend API Configuration ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

# --- Sidebar ---
with st.sidebar:
    st.header("About SophiaWeaver")
    st.markdown("""
    Welcome to SophiaWeaver! 🕊️

    This is your friendly AI companion for exploring the Bible.
    Ask questions, seek understanding, and discover insights from the scriptures.

    **Features (Coming Soon):**
    *   Save and revisit conversations
    *   Personalized reading plans
    *   Verse of the day

    We hope you find this tool helpful on your spiritual journey!
    """)
    st.markdown("---")
    st.caption("© 2024 SophiaWeaver Project")


# --- App Title ---
st.title("SophiaWeaver 🕊️ - Your Bible Chat Companion")
st.caption("Ask, explore, and learn about the Bible in a friendly way!")

# --- Initialize Chat History ---
# This will be a single conversation for now
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! 👋 I'm SophiaWeaver. How can I help you explore the Bible today?"}
    ]

# --- Display Chat Messages ---
for message in st.session_state.messages:
    avatar_emoji = "👤" if message["role"] == "user" else "📖" # User and Bot Emojis
    with st.chat_message(message["role"], avatar=avatar_emoji):
        st.markdown(message["content"])

# --- Accept User Input ---
prompt = st.chat_input("Type your Bible question here... 🤔")

if prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant", avatar="📖"):
        message_placeholder = st.empty()
        # Simulate thinking with a placeholder and emoji
        with message_placeholder.container():
            st.markdown("Thinking... 🧠")
            # You can add a spinner too:
            # with st.spinner("Sophia is searching the scriptures..."):
            #     time.sleep(0.5) # Small delay for effect, remove if backend is very fast

        full_response_text = ""
        try:
            payload = {
                "user_input": prompt,
                "max_length": 150,
                "temperature": 0.7
            }
            response: Response = requests.post(CHAT_ENDPOINT, json=payload, timeout=30) # Added timeout
            response.raise_for_status()

            bot_response_data = response.json()
            full_response_text = bot_response_data.get("bot_response", "Hmm, I'm not sure how to answer that right now. 😅")

        except requests.exceptions.Timeout:
            st.error("⏳ Oops! The request timed out. The server might be busy. Please try again.")
            full_response_text = "Sorry, I took too long to respond. Could you try asking again?"
        except requests.exceptions.RequestException as e:
            st.error(f"🔗 Oh no! I couldn't connect to my brain (the backend): {e}")
            full_response_text = "I'm having a little trouble connecting right now. Please check back soon!"
        except Exception as e:
            st.error(f"💥 Yikes! An unexpected glitch happened: {e}")
            full_response_text = "Something unexpected went wrong. My apologies!"

        # Update the placeholder with the actual response
        message_placeholder.markdown(full_response_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response_text})

# --- Footer (Optional) ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Powered by SophiaWeaver AI</p>", unsafe_allow_html=True)