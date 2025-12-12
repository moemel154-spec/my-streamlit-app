import streamlit as st
import pandas as pd
import json
import re
import google.generativeai as genai

st.set_page_config(page_title="Lite Vocabulary Extractor", layout="wide")

# ---------------- API KEY ----------------
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

MODEL = "models/gemini-2.5-flash"

# ---------------- SIMPLE CALL ----------------
def ask(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL)
        r = model.generate_content(prompt)
        return r.text or ""
    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- JSON CLEAN ----------------
def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return m.group(0) if m else "[]"

# ---------------- UI ----------------
st.title("📘 Ultra-Lite Vocabulary Extractor")
novel = st.text_input("Novel Title", "")

if st.button("Analyze") and novel.strip():
    if not api_key:
        st.error("Enter API key first.")
        st.stop()

    st.info("Running only 2 API calls. Please wait...")

    # ============ CALL 1 ============
    prompt1 = f"""
