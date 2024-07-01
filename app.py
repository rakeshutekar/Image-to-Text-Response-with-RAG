import os

import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from models.blip_model import extract_image_details
from models.gpt_model import generate_response

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize session state for image details
if "image_details" not in st.session_state:
    st.session_state.image_details = ""

st.title("Image to Text Response with RAG")

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if "uploaded_file" not in st.session_state or uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        # Clear previous messages
        st.session_state.messages = []

        # Show loading status while processing the image
        with st.status("Processing image...", state="running"):
            image = Image.open(uploaded_file)
            st.session_state.image_details = extract_image_details(image)

        st.success("Image processed successfully.")
else:
    st.stop()

# Chat interface
if st.session_state.image_details:
    st.write("### Ask a question about the image")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    prompt = st.chat_input("Ask something about the image")
    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        response = generate_response([st.session_state.image_details], prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant message
        with st.chat_message("assistant"):
            st.write(response)
