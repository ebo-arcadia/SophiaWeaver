# SophiaWeaver - GenAI Chatbot

Welcome to **SophiaWeaver**! This project aims to create a versatile GenAI chatbot application, starting with a focus on specific knowledge domains like The Bible, and designed for future expansion into other areas like sociology and psychology, with capabilities for inter-model interactions.

This README focuses on **Phase 1: Bare Minimum MVP**.

## Project Structure

-   `SophiaWeaver/` (Project Root)
    -   `app/`: Contains the backend (FastAPI) and frontend (Streamlit) applications.
    -   `data/`:
        -   `content_sources/domain_the_bible/`: Original PDF content (e.g., books of The Bible).
        -   `processed_texts/domain_the_bible/`: Processed text data from The Bible.
    -   `trained_models/the_bible_gpt2_small/`: Stores the fine-tuned model for The Bible.
    -   `scripts/`: Utility scripts for data processing and model training.
    -   `notebooks/`: Jupyter notebooks for experimentation.

## Phase 1 Goals

-   Simple FastAPI backend server.
-   Streamlit frontend UI for chat.
-   Fine-tune a small GPT-2 model on a few PDF files related to The Bible.
-   Containerize using Docker.
-   (Conceptual) Deployment on Hugging Face Spaces.

## Prerequisites

-   Python 3.8+
-   Docker & Docker Compose (optional, for local multi-container setup)
-   Access to a GPU is highly recommended for model fine-tuning.
-   Git

## Setup and Installation

1.  **Clone the repository:**
    