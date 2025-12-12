import streamlit as st
import pandas as pd
import json
import re
import google.generativeai as genai
import io

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Intelligent Vocabulary Extractor", layout="wide", initial_sidebar_state="expanded")

# ----------------- STYLES (Pastel Mint + Prompt font) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');

/* Root colors */
:root{
  --bg: #eefef6;
  --card: #ffffff;
  --accent-start: #a6f3d1;
  --accent-end: #7bd3b0;
  --mark: #e8fff5;
  --text-dark: #013826;
}

/* Global */
html, body, [class*="css"] {
  font-family: 'Prompt', sans-serif;
  background: linear-gradient(180deg, #f6fff9 0%, #eefef6 100%) !important;
}

/* Header with animated gradient */
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
  box-shadow: 0 6px 18px rgba(95,192,154,0.12);
  margin-bottom: 10px;
  transform: translateZ(0);
}

/* subtle floating */
@keyframes floatUp {
  from { transform: translateY(6px); }
  to { transform: translateY(0); }
}
.app-header { animation: gradientShift 8s ease infinite, floatUp 0.8s ease backwards; }

.header-title { font-size: 26px; font-weight:700; margin-bottom:2px; }
.header-sub { font-size:12px; opacity:0.95; }

/* Sections & cards — fade-in */
.section-header {
  font-size:17px;
  font-weight:700;
  color: var(--text-dark);
  margin: 4px 0 6px 0 !important;
}
.card {
  background: var(--card);
  padding: 12px 14px;
  border-radius: 10px;
  box-shadow: 0 3px 10px rgba(7,57,42,0.05);
  border-left: 3px solid rgba(123,211,176,0.55);
  margin-bottom: 10px;
  animation: floatUp 0.35s ease backwards;
}

/* tighter input */
.stTextInput>div>div>input {
  padding: 8px !important;
  font-size: 15px !important;
  border-radius: 8px !important;
  border: 1px solid #cfeee2 !important;
}

/* Highlight */
mark {
  background-color: var(--mark);
  padding: 3px 5px;
  border-radius: 6px;
  font-weight:700;
}

/* Warning pill (animated pulse) */
.warning-pill {
  display:inline-block;
  background: linear-gradient(90deg, rgba(255,230,230,0.9), rgba(255,250,230,0.9));
  color: #7a2618;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight:700;
  border: 1px solid rgba(250,180,160,0.6);
  box-shadow: 0 6px 18px rgba(250,160,140,0.06);
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(123,211,176,0.12); }
  70% { box-shadow: 0 0 18px 6px rgba(123,211,176,0.04); }
  100% { box-shadow: 0 0 0 0 rgba(123,211,176,0); }
}
.warning-pill { animation: pulse 2.8s infinite; }

/* Button */
.stButton>button {
  background: linear-gradient(90deg,var(--accent-start),var(--accent-end));
  color: #013826;
  font-weight:700;
  border-radius:8px;
  padding:8px 14px;
  border: none;
}

/* reduce df padding */
[data-testid="stDataFrameContainer"] { padding: 6px !important; }

</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.header("🔑 API")
gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.caption("This app searches public web snippets for reviews and example sentences.")
genai.configure(api_key=gemini_key)

# ----------------- HELPERS -----------------
def extract_json_array(text: str) -> str:
    """Return the first JSON array substring found in text, cleaned of markdown and smart quotes."""
    if not text:
        return "[]"
    # remove markdown fences if present
    text = text.replace("```json", "").replace("```", "")
    # normalize quotes
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    # extract [...], greedy by default but DOTALL used
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        return m.group(0)
    # fallback: try to find {...} objects sequence
    return text

def ask_gemini(prompt: str, web: bool = False) -> str:
    """Call Gemini model; if web=True, enable web_search tool."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        if web:
            response = model.generate_content(
                prompt,
                tools={"web_search": {}},
                tool_config={"web_search": {"n_tokens": 4096}},
            )
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

def repair_json_with_model(bad_text: str) -> str:
    """Ask Gemini to fix invalid JSON; return cleaned JSON string or original."""
    repair_prompt = f"""
The following text is intended to be a JSON array but may be malformed. Fix it and return ONLY valid JSON (the array). No explanations.
{bad_text}
"""
    repaired = ask_gemini(repair_prompt, web=False)
    return extract_json_array(repaired)

# ----------------- APP HEADER -----------------
st.markdown("""
<div class="app-header">
  <div style="display:flex; gap:16px; align-items:center;">
      <div class="header-title">📘 Vocabulary Extractor for English Beginners </div>
      <div class="header-sub"> Analyze a Novel Title • Extract Hard Words • Summarize Reading Difficulty</div>
""", unsafe_allow_html=True)

# ----------------- INPUT -----------------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔎Novel Input</div>', unsafe_allow_html=True)
st.markdown("Tip: Pick popular or critically acclaim novels for better results.", unsafe_allow_html=True)
novel_title = st.text_input(
    "", placeholder="Type your novel title here (ex. 1984, The Great Gatsby, Pride and Prejudice)",)
analyze_btn = st.button("Analyze Novel", help="Click Here")

# ----------------- FUNCTIONS FOR TASKS -----------------
def get_difficulty_summary(novel_title: str) -> pd.DataFrame:
    """Always use web search to gather snippets, then summarize into aspects (JSON)."""
    # web search step
    web_prompt = f"Search public web snippets and collect short reader opinions about how difficult '{novel_title}' is to read. Return collected short snippets only."
    web_text = ask_gemini(web_prompt, web=True)
    # fallback to internal knowledge if web failed
    if not web_text or web_text.startswith("❌ Error") or web_text.strip() == "":
        web_text = ask_gemini(f"Summarize common public reader opinions about why '{novel_title}' may be difficult to read. Use public knowledge.", web=False)
    summary_prompt = f"""
Below are web-snippets about reading difficulty for the novel "{novel_title}". Extract 5 common difficulty aspects and return a JSON array EXACTLY like:
[
  {{"Aspect": "Vocabulary", "Summary": "one brief sentence"}},
  {{"Aspect": "Syntax", "Summary": "one brief sentence"}},
  {{"Aspect": "Themes", "Summary": "one brief sentence"}},
  {{"Aspect": "Cultural context", "Summary": "one brief sentence"}}
  {{"Aspect": "Content warning", "Summary": "one brief sentence"}}
]
Do NOT provide explanations, reasoning, or method descriptions. Do NOT copy verbatim; paraphrase.
Web snippets:
{web_text}
"""
    raw = ask_gemini(summary_prompt, web=False)
    cleaned = extract_json_array(raw)
    # try parse, else repair once
    try:
        data = json.loads(cleaned)
        return pd.DataFrame(data)
    except Exception:
        repaired = repair_json_with_model(raw)
        try:
            data = json.loads(repaired)
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame([{"aspect": "Error", "summary": "No web data or parse failed"}])

def generate_sentences(novel_title: str) -> str:
    p = f"Copy 10 original short, distinct and crucial sentences from '{novel_title}'. Choose the most popular if possible. Do NOT explain any of your process. JUST RETURN THE SENTENCES ONLY."
    return ask_gemini(p, web=False)

def extract_vocab(sentences: str, n: int = 20):
    prompt = f"""
Extract {n} English vocabulary items from the sentences below.

RULES:
- The word must come DIRECTLY from the sentences.
- Use the exact word form (no lemmatization).
- Output fields per item:
    - word
    - meaning_en
    - part_of_speech 
    - CEF_level (A1, A2, B1, B2, C1, C2) 
    - example_sentence 

STRICT:
- JSON array ONLY
- NO explanations. NO breakdowns. NO comments.

Sentences:
{sentences}
"""
    raw = ask_gemini(prompt, web=False)
    cleaned = extract_json_array(raw)
    try:
        arr = json.loads(cleaned)
    except:
        repaired = repair_json_with_model(raw)
        try:
            arr = json.loads(repaired)
        except:
            return None, raw
    for o in arr:
        w = o.get("word", "")
        w = re.sub(r"\[.*?\]", "", w).strip()
        o["word"] = w
    cef_order = {"C2": 6, "C1": 5, "B2": 4, "B1": 3, "A2": 2, "A1": 1}
    arr = sorted(arr, key=lambda x: cef_order.get(x.get("CEF_level", "B1"), 3), reverse=True)
    return arr, raw

# ----------------- RUN ANALYSIS -----------------
if analyze_btn and novel_title.strip() != "":
    if not gemini_key:
        st.error("Please enter your Google Gemini API Key in the sidebar.")
        st.stop()

    with st.spinner("This process may take a few seconds."):
        # 1) Summary
        summary_prompt = f"""
Summarize the novel '{novel_title}' in exactly 6 concise sentences.
- No spoilers.
- Do NOT explain your reasoning, give warnings, or include any breakdown.
- Return ONLY the 6 sentences as plain text (no JSON, no headings).
"""
        summary = ask_gemini(summary_prompt, web=False)

        # 2) Difficulty reviews / summary (web)
        df_reviews = get_difficulty_summary(novel_title)

        # 3) Example sentences
        sentences = generate_sentences(novel_title)

        # 4) Extract vocab (n=10 as your code did)
        vocab_list, raw_vocab = extract_vocab(sentences, n=10)
        if vocab_list is None:
            vocab_list = []
            df_vocab = pd.DataFrame()
        else:
            df_vocab = pd.DataFrame(vocab_list)

        # 5) Highlight sentences — APPLY substitution INSIDE loop for each word
        highlighted_sentences = sentences if isinstance(sentences, str) else str(sentences)
        for it in vocab_list:
            w = it.get("word", "")
            if not w:
                continue
            # compile pattern to match whole word, case-insensitive
            pattern = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
            highlighted_sentences = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted_sentences)

    # --- now display in tabs (example) ---
    tab1, tab2, tab3 = st.tabs(["Overview","Vocabulary","Download"])

    with tab1:
        st.markdown("""
        <div class="card" style="
        background:rgba(255,255,255,0.55);
        backdrop-filter:blur(12px);
        border-left:4px solid #78d8b2;
        box-shadow:0 8px 22px rgba(0,0,0,0.07);
        animation: floatUp 0.4s ease forwards;
        ">
        <div style="font-size:20px; font-weight:700; color:#014f39; margin-bottom:8px;">
          🌿 Novel Summary
        </div>
        <div style="font-size:14px; line-height:1.6; color:#013826;">
        """, unsafe_allow_html=True)

        # Generate enriched summary
        enrich_prompt = f"""
        Improve the following 6-sentence summary of '{novel_title}' into a more detailed,
        elegant and descriptive summary — but still NO spoilers.
        Add:
        - Tone & mood of the book
        - Writing style (but no spoilers)
        - Suitable reader profile
        - Overall atmosphere

        DO NOT output headings.
        Output as 2 detailed paragraphs (6–8 sentences total).
        Summary to improve:
        {summary}
        """
        enriched_summary = ask_gemini(enrich_prompt, web=False)
        st.write(enriched_summary)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # --- READING DIFFICULTY REPORT ---
        st.markdown("""
        
        <div class="card" style="
        background:rgba(255,255,255,0.58);
        border-left:4px solid #ffa07a;
        backdrop-filter:blur(12px);
        ">
        <div style="font-size:18px; font-weight:700; color:#5a1a00; margin-bottom:4px;">
            🔎 Readability Insights
        </div>
        <div style="color:#4b2d20; font-size:14px; margin-bottom:10px;">
            Summarized from web reader opinions across multiple public sources.
        </div>
        """, unsafe_allow_html=True)

        # create enhanced difficulty analysis
        review_prompt = f"""
        From the following difficulty-aspect table, create an improved readability analysis.
         Include:
        - CEFR difficulty guess (A1–C2)
        - Biggest challenges for readers
        - What type of readers enjoy this style
        - Sentence complexity overview
        - A short recommendation sentence

         DO NOT show headings. Output 2 short paragraphs.

        Table:
        {df_reviews.to_dict(orient='records')}
         """
        review_insights = ask_gemini(review_prompt, web=False)
        st.write(review_insights)

        st.markdown("""
        <div style="margin-top:12px;">
        """, unsafe_allow_html=True)
        st.dataframe(df_reviews)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with tab2:
    # ----- Example Sentences -----
        st.markdown("""
    <div class="section-header" style="margin-top:10px;">✨ Highlighted Example Sentences</div>
    """, unsafe_allow_html=True)

        st.markdown(highlighted_sentences.replace("\n\n", "<br><br>").replace("\n", "<br>"),
                unsafe_allow_html=True)

        st.markdown("""
    <div class="section-header" style="margin-top:16px;">📚 Vocabulary Table</div>
    """, unsafe_allow_html=True)

        if not df_vocab.empty:
            st.dataframe(df_vocab, use_container_width=True)
        else:
            st.info("No vocabulary data available. Please try again.")

    with tab3:
        st.markdown("""
    <div class="section-header" style="margin-top:12px;">📥 Download Your Data</div>
    """, unsafe_allow_html=True)

        if not df_vocab.empty:
            csv = df_vocab.to_csv(index=False).encode('utf-8')
            st.download_button(
            "📄 Download CSV (Vocab)",
                csv,
            "vocab.csv",
            "text/csv",
            use_container_width=True,
        )
        else:
            st.info("No data to download yet. Please analyze a novel first.")

    st.markdown("</div></div>", unsafe_allow_html=True)





    







