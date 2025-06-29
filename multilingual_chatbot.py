# app.py
import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)
from googletrans import Translator
from langdetect import detect
from gtts import gTTS
import torch
import uuid
import os
import logging

st.set_page_config(page_title="🤖 Smart Multilingual Chatbot", layout="centered")
st.title("🤖 Smart Multilingual Chatbot")
st.caption("Supports multiple models with voice & translation!")

# Logging setup
logging.basicConfig(level=logging.ERROR)

# Load DialoGPT model
@st.cache_resource
def load_dialo():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

# Load BlenderBot model safely
@st.cache_resource
def load_blender():
    try:
        model_name = "facebook/blenderbot-400M-distill"  # smaller and faster model
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logging.error(f"BlenderBot loading failed: {e}")
        st.error("⚠️ Failed to load BlenderBot model. Please check your internet or try again later.")
        return None, None

translator = Translator()

# Model selection
model_option = st.selectbox("Choose a model", ["DialoGPT (English only)", "BlenderBot (Multilingual)"])

tokenizer, model = (load_dialo() if model_option == "DialoGPT (English only)" else load_blender())

# Chat history
if "past" not in st.session_state:
    st.session_state.past = []
if "bot" not in st.session_state:
    st.session_state.bot = []

# User input
user_input = st.text_input("💬 Say something in any language:")

# Text-to-speech helper
def speak_text(text, lang):
    tts = gTTS(text=text, lang=lang)
    filename = f"voice_{uuid.uuid4()}.mp3"
    tts.save(filename)
    return filename

# GIFs based on emotion
def get_mood_response(text):
    if any(word in text.lower() for word in ["hello", "hi"]):
        return "👋", "https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif"
    elif any(word in text.lower() for word in ["happy", "great"]):
        return "😄", "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif"
    elif any(word in text.lower() for word in ["sad", "bad"]):
        return "😢", "https://media.giphy.com/media/3og0IPxMM0erATueVW/giphy.gif"
    return "🤖", "https://media.giphy.com/media/26FPy3QZQqGtDcrja/giphy.gif"

# Generate response on button click
if st.button("🚀 Send") and user_input:
    if tokenizer is None or model is None:
        st.warning("Model not available. Please select another or retry.")
    else:
        try:
            user_lang = detect(user_input)
            translated_input = translator.translate(user_input, dest='en').text

            if model_option == "DialoGPT (English only)":
                input_ids = tokenizer.encode(translated_input + tokenizer.eos_token, return_tensors='pt')
                output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            else:
                inputs = tokenizer(translated_input, return_tensors="pt")
                result = model.generate(**inputs)
                response = tokenizer.decode(result[0], skip_special_tokens=True)

            translated_output = translator.translate(response, dest=user_lang).text
            emoji, gif = get_mood_response(translated_output)

            st.session_state.past.append(user_input)
            st.session_state.bot.append(f"{emoji} {translated_output}")

            audio_file = speak_text(translated_output, user_lang)
            st.audio(audio_file)
            st.image(gif, width=300)

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

# Display chat history
if st.session_state.past:
    st.subheader("🧠 Chat History")
    for user, bot in zip(st.session_state.past, st.session_state.bot):
        st.markdown(f"👤 **You**: {user}")
        st.markdown(bot)
