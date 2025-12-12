import streamlit as st
import pandas as pd
import json
import re
import google.generativeai as genai

st.set_page_config(page_title="Vocabulary Extractor", layout="wide")

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
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"

# ---------------- UI ----------------
st.title("📘Vocabulary Extractor")
novel = st.text_input("Novel Title", "")

if st.button("Analyze") and novel.strip():
    if not api_key:
        st.error("Enter API key first.")
        st.stop()

    # ============ CALL 1 ============

    prompt1 = (
        "For the novel \"" + novel + "\" return EXACTLY the following JSON object:\n\n"
        "{\n"
        "  \"summary\": \"6-sentence spoiler-free summary.\",\n"
        "  \"difficulty\": [\n"
        "     {\"Aspect\": \"Vocabulary\", \"Summary\": \"one sentence\"},\n"
        "     {\"Aspect\": \"Syntax\", \"Summary\": \"one sentence\"},\n"
        "     {\"Aspect\": \"Themes\", \"Summary\": \"one sentence\"},\n"
        "     {\"Aspect\": \"Cultural context\", \"Summary\": \"one sentence\"},\n"
        "     {\"Aspect\": \"Content warning\", \"Summary\": \"one sentence\"}\n"
        "  ],\n"
        "  \"sentences\": [\n"
        "     \"10 short important example sentences from the novel\"\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- NO extra commentary\n"
        "- NO explanations\n"
        "- ONLY output valid JSON\n"
    )

    raw1 = ask(prompt1)

    try:
        data1 = json.loads(extract_json(raw1))
    except Exception:
        st.error("JSON parse failed.")
        st.text(raw1)
        st.stop()

    summary = data1.get("summary", "")
    diff = pd.DataFrame(data1.get("difficulty", []))
    example_sentences = data1.get("sentences", [])

    # ============ CALL 2 ============

    sentences_text = "\n".join(example_sentences)

    prompt2 = (
        "Extract 12 vocabulary items from the sentences below. "
        "Use EXACT words from the sentences.\n\n"
        "Output ONLY a pure JSON array:\n"
        "[\n"
        "  {\n"
        "    \"word\": \"\",\n"
        "    \"meaning_en\": \"\",\n"
        "    \"part_of_speech\": \"\",\n"
        "    \"CEF_level\": \"A1/A2/B1/B2/C1/C2\",\n"
        "    \"example_sentence\": \"\"\n"
        "  }\n"
        "]\n\n"
        "Sentences:\n" + sentences_text
    )

    raw2 = ask(prompt2)

    try:
        vocab = json.loads(extract_json(raw2))
    except Exception:
        vocab = []
        st.warning("Vocabulary JSON parse failed.")

    df_vocab = pd.DataFrame(vocab)

    # ============ OUTPUT ============

    st.subheader("📖 Summary")
    st.write(summary)

    st.subheader("📊 Reading Difficulty")
    st.dataframe(diff, use_container_width=True)

    st.subheader("📘 Example Sentences")
    for s in example_sentences:
        st.markdown(f"- {s}")

    st.subheader("📚 Vocabulary")
    if not df_vocab.empty:
        st.dataframe(df_vocab, use_container_width=True)
        st.download_button(
            "Download CSV",
            df_vocab.to_csv(index=False).encode(),
            "vocab.csv",
            "text/csv"
        )
    else:
        st.info("No vocabulary available.")


