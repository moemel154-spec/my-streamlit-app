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
        response = model.generate_content(prompt)
        return response.text or ""
    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- JSON CLEAN ----------------
def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"

# ---------------- UI ----------------
st.title("📘 Ultra-Lite Vocabulary Extractor")
novel = st.text_input("Novel Title", "")

if st.button("Analyze") and novel.strip():
    if not api_key:
        st.error("Enter API key first.")
        st.stop()

    st.info("Running only 2 API calls…")

    # ============ CALL 1 ============
    prompt1 = (
        "For the novel \"" + novel + "\" return EXACTLY the following JSON object:\n\n"
        "{\n"
        "  \"summary\": \"6-sentence spoiler-free summary.\",\n"
        "  \"difficulty\": [\n"
        "     {\"Aspect\": \"Vocabulary\", \"Summary\": \"one sentence\"},\n"
        "     {\"Aspect\"
