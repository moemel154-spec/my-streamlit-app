import streamlit as st
import pandas as pd
import json
import re
import time
import google.generativeai as genai
import io

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Intelligent Vocabulary Extractor", layout="wide", initial_sidebar_state="expanded")

# ----------------- STYLES (Pastel Mint + Prompt font) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');

:root{
  --bg: #eefef6;
  --card: #ffffff;
  --accent-start: #a6f3d1;
  --accent-end: #7bd3b0;
  --mark: #e8fff5;
  --text-dark: #013826;
}

html, body, [class*="css"] {
  font-family: 'Prompt', sans-serif;
  background: linear-gradient(180deg, #f6fff9 0%, #eefef6 100%) !important;
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.app-header {
  padding: 16px 22px;
  border-radius: 10px;
  background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
  background-size: 200% 200%;
  animation: gradientShift 8s ease infinite;
  color: white;
  margin-bottom: 10px;
}

mark {
  background-color: var(--mark);
  padding: 3px 5px;
  border-radius: 6px;
  font-weight:700;
}

.stButton>button {
  background: linear-gradient(90deg,var(--accent-start),var(--accent-end));
  color: #013826;
  font-weight:700;
  border-radius:8px;
  padding:8px 14px;
  border: none;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.header("🔑 API")
gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.caption("This app searches public web snippets for reviews and example sentences.")
genai.configure(api_key=gemini_key)

# ----------------- SAFE CALL (prevent rate limit) -----------------
def safe_call(prompt: str, web: bool = False) -> str:
    time.sleep(1.2)  # prevent 429
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        if web:
            response = model.generate_content(
                prompt,
                tools={"web_search": {}},
                tool_config={"web_search": {"n_tokens": 2048}},
            )
        else:
            response = model.generate_content(prompt)
        return response.text or ""
    except Exception as e:
        return f"❌ Error: {e}"

# ----------------- HELPERS -----------------
def extract_json_array(text: str) -> str:
    if not text:
        return "[]"
    text = text.replace("```json", "").replace("```", "")
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    m = re.search(r'\[.*\]', text, re.DOTALL)
    return m.group(0) if m else "[]"

# ----------------- MAIN FUNCTIONS -----------------
def fetch_web_difficulty(novel_title: str) -> pd.DataFrame:
    prompt = f"""
Search public web snippets and extract 5 difficulty aspects for the novel '{novel_title}'.
Return JSON ONLY:
[
  {{"Aspect": "Vocabulary", "Summary": "..."}},
  {{"Aspect": "Syntax", "Summary": "..."}},
  {{"Aspect": "Themes", "Summary": "..."}},
  {{"Aspect": "Cultural context", "Summary": "..."}},
  {{"Aspect": "Content warning", "Summary": "..."}}
]
"""
    raw = safe_call(prompt, web=True)
    data = json.loads(extract_json_array(raw))
    return pd.DataFrame(data)

def get_sentences_and_vocab(novel_title: str):
    prompt = f"""
For the novel '{novel_title}':

1) Provide EXACTLY 8 important short sentences from the book (not too long).
2) From those sentences, extract EXACTLY 10 vocabulary items with:
   - word
   - meaning_en
   - part_of_speech
   - CEF_level
   - example_sentence

Return JSON ONLY as:
{{
  "sentences": ["...", "..."],
  "vocab": [
     {{"word":"...", "meaning_en":"...", "part_of_speech":"...", "CEF_level":"...", "example_sentence":"..."}}
  ]
}}
"""
    raw = safe_call(prompt, web=False)

    try:
        cleaned = extract_json_array(raw)
        obj = json.loads(cleaned)
    except:
        return [], []

    sentences = obj.get("sentences", [])
    vocab = obj.get("vocab", [])

    cef_order = {"C2":6, "C1":5, "B2":4, "B1":3, "A2":2, "A1":1}
    vocab = sorted(vocab, key=lambda x: cef_order.get(x.get("CEF_level","B1"),3), reverse=True)

    return sentences, vocab

def get_summary(novel_title: str):
    prompt = f"""
Write a 2-paragraph descriptive summary of '{novel_title}'.
Include:
- Tone and atmosphere
- Writing style
- Suitable reader profile
- No spoilers

8–10 sentences total. No headings.
"""
    return safe_call(prompt)

# ----------------- UI ELEMENTS -----------------
st.markdown("""
<div class="app-header">
  <div class="header-title">📘 Vocabulary Extractor for English Beginners</div>
  <div class="header-sub">Analyze a Novel • Extract Words • Get Readability Insights</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔎 Novel Input")
novel_title = st.text_input("", placeholder="Example: 1984, The Great Gatsby")
analyze_btn = st.button("Analyze Novel")

# ----------------- ANALYSIS PIPELINE -----------------
if analyze_btn and novel_title.strip() != "":
    if not gemini_key:
        st.error("Please enter your Google Gemini API Key.")
        st.stop()

    with st.spinner("Processing…"):
        summary = get_summary(novel_title)
        df_reviews = fetch_web_difficulty(novel_title)
        sentences, vocab = get_sentences_and_vocab(novel_title)

    tab1, tab2, tab3 = st.tabs(["Overview", "Vocabulary", "Download"])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("🌿 Enhanced Summary")
        st.write(summary)

        st.subheader("🔎 Readability Insights")
        st.dataframe(df_reviews)

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("✨ Example Sentences (Highlighted)")
        text_block = "\n".join(sentences)
        for item in vocab:
            w = item.get("word", "")
            if w:
                pattern = re.compile(rf"\\b{re.escape(w)}\\b", flags=re.IGNORECASE)
                text_block = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text_block)

        st.markdown(text_block.replace("\n", "<br>"), unsafe_allow_html=True)

        st.subheader("📚 Vocabulary Table")
        df_vocab = pd.DataFrame(vocab)
        st.dataframe(df_vocab, use_container_width=True)

    # ---------- TAB 3 ----------
    with tab3:
        if vocab:
            csv = pd.DataFrame(vocab).to_csv(index=False).encode("utf-8")
            st.download_button("📄 Download Vocabulary CSV", csv, "vocab.csv", "text/csv")
        else:
            st.info("No data to download yet.")

