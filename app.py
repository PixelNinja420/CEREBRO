import streamlit as st
import pandas as pd
import re
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="CEREBRO: Tone Intelligence", layout="wide")
st.title("🧠 CEREBRO // Neural Sentiment Engine")
st.caption("Multi-turn chat sentiment & tone tracking · Sarcasm & passive-aggression detection · Emotion arc visualisation")

# --- LOAD ML MODELS (cached so they only load once) ---
@st.cache_resource
def load_models():
    from transformers import pipeline
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    emotion = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None
    )
    return sentiment, emotion

with st.spinner("Loading neural models... (first run takes ~30s)"):
    sentiment_model, emotion_model = load_models()

# --- REGEX PATTERN DETECTORS ---
PATTERNS = {
    "blame": [r"\byou always\b", r"\byou never\b", r"\bbecause of you\b", r"\byour fault\b"],
    "hostility": [r"\bidiot\b", r"\bstupid\b", r"\bshut up\b", r"\bhate\b", r"\bcan't stand\b"],
    "defensiveness": [r"\bthat's not true\b", r"\bi didn't\b", r"\bnot my fault\b", r"\bdon't blame me\b"],
    "withdrawal": [r"\bfine\b", r"\bwhatever\b", r"\bforget it\b", r"\bnever mind\b", r"\bi'm done\b", r"\bpointless\b"],
    "passive_aggression": [r"\bper my last\b", r"\bas i already said\b", r"\btake all the time you need\b",
                           r"\bi guess i'll just\b", r"\bno worries\b.*\bat all\b", r"\bsure\b.*\bwhatever you say\b"],
    "sarcasm_hint": [r"\boh great\b", r"\bwow\b.*\breally\b", r"\byeah right\b", r"\bsure thing\b",
                     r"\bbrilliant\b", r"\bfantastic\b.*\bjob\b", r"\bthanks for nothing\b"]
}

def detect_patterns(text):
    text_lower = text.lower()
    scores = {}
    for category, pattern_list in PATTERNS.items():
        scores[category] = int(any(re.search(p, text_lower) for p in pattern_list))
    return scores

# --- ANALYSIS FUNCTIONS ---
def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result["label"], result["score"]

def analyze_emotions(text):
    results = emotion_model(text[:512])[0]
    return {r["label"]: round(r["score"], 4) for r in results}

def compute_tension_score(emotion_scores, pattern_scores):
    tension = (
        0.25 * emotion_scores.get("anger", 0) +
        0.15 * emotion_scores.get("annoyance", 0) +
        0.12 * emotion_scores.get("disapproval", 0) +
        0.10 * emotion_scores.get("disgust", 0) +
        0.08 * emotion_scores.get("sadness", 0) +
        0.10 * pattern_scores.get("blame", 0) +
        0.08 * pattern_scores.get("hostility", 0) +
        0.05 * pattern_scores.get("passive_aggression", 0) +
        0.04 * pattern_scores.get("sarcasm_hint", 0) +
        0.03 * pattern_scores.get("withdrawal", 0)
    )
    return round(min(max(tension, 0), 1), 4)

def compute_warmth_score(emotion_scores):
    warmth = (
        0.35 * emotion_scores.get("joy", 0) +
        0.20 * emotion_scores.get("love", 0) +
        0.15 * emotion_scores.get("admiration", 0) +
        0.10 * emotion_scores.get("gratitude", 0) +
        0.10 * emotion_scores.get("caring", 0) +
        0.10 * emotion_scores.get("optimism", 0)
    )
    return round(min(max(warmth, 0), 1), 4)

# --- PARSING FUNCTION ---
def parse_chat_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    lines = content.split('\n')
    parsed_data = []
    turn = 1
    for line in lines:
        if not line.strip():
            continue
        match = re.search(r'(?:\[.*?\]\s*)?([^:]+):\s*(.*)', line)
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
            if message:
                parsed_data.append({"turn": turn, "speaker": speaker, "message": message})
                turn += 1
    return pd.DataFrame(parsed_data)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("Upload a chat export from **WhatsApp**, **Slack**, **Discord**, or any messenger.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported formats:**")
st.sidebar.code("Name: Message\n[Time] Name: Message", language="text")

# --- MAIN UI ---
uploaded_file = st.file_uploader("📂 Upload a Chat Log (.txt)", type=["txt"])

if uploaded_file is not None:
    df = parse_chat_file(uploaded_file)

    if df.empty:
        st.error("Could not parse any messages. Make sure the format is `Name: Message`.")
    else:
        st.success(f"✅ Parsed **{len(df)} messages** from **{df['speaker'].nunique()} participants**.")

        # --- RUN ANALYSIS ---
        progress = st.progress(0, text="Analyzing tone...")
        sentiments, emotions_list, tensions, warmths = [], [], [], []
        all_patterns = []

        for i, row in df.iterrows():
            msg = row["message"]

            # Sentiment
            label, score = analyze_sentiment(msg)
            sentiments.append(label)

            # Emotions
            emo = analyze_emotions(msg)
            emotions_list.append(emo)

            # Patterns
            pats = detect_patterns(msg)
            all_patterns.append(pats)

            # Scores
            tensions.append(compute_tension_score(emo, pats))
            warmths.append(compute_warmth_score(emo))

            progress.progress((i + 1) / len(df), text=f"Analyzing message {i+1}/{len(df)}...")

        progress.empty()

        df["sentiment"] = sentiments
        df["tension"] = tensions
        df["warmth"] = warmths

        # Rolling averages for smoother arc
        df["tension_trend"] = df["tension"].rolling(window=3, min_periods=1).mean().round(4)
        df["warmth_trend"] = df["warmth"].rolling(window=3, min_periods=1).mean().round(4)

        # Escalation detection
        df["escalation"] = df["tension_trend"].diff().fillna(0)
        df["is_escalating"] = df["escalation"] > 0.02

        # Add pattern columns
        pattern_df = pd.DataFrame(all_patterns)
        df = pd.concat([df, pattern_df], axis=1)

        # Top emotion per message
        df["dominant_emotion"] = [max(e, key=e.get) for e in emotions_list]

        # --- DISPLAY: METRICS ---
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💬 Messages", len(df))
        col2.metric("🔥 Peak Tension", f"{df['tension'].max():.2f}")
        col3.metric("💚 Peak Warmth", f"{df['warmth'].max():.2f}")
        col4.metric("⚠️ Escalation Points", int(df["is_escalating"].sum()))

        # --- DISPLAY: EMOTION ARC ---
        st.markdown("---")
        st.subheader("📈 Emotion Arc")

        arc_df = df[["turn", "tension", "warmth", "tension_trend", "warmth_trend"]].set_index("turn")
        st.line_chart(arc_df[["tension_trend", "warmth_trend"]], color=["#ff4b4b", "#4bff88"])

        # --- DISPLAY: DETAILED ARC ---
        with st.expander("🔍 Raw Tension & Warmth (per message)"):
            st.line_chart(arc_df[["tension", "warmth"]], color=["#ff6b6b", "#6bffa8"])

        # --- DISPLAY: SPEAKER ANALYSIS ---
        st.markdown("---")
        st.subheader("👥 Speaker Analysis")
        speaker_stats = df.groupby("speaker").agg(
            avg_tension=("tension", "mean"),
            max_tension=("tension", "max"),
            avg_warmth=("warmth", "mean"),
            messages=("turn", "count")
        ).round(3).sort_values("avg_tension", ascending=False)
        st.dataframe(speaker_stats, use_container_width=True)

        # --- DISPLAY: FLAGGED MESSAGES ---
        st.markdown("---")
        st.subheader("🚩 Flagged Messages")

        flag_cols = ["blame", "hostility", "defensiveness", "withdrawal", "passive_aggression", "sarcasm_hint"]
        df["flags"] = df[flag_cols].sum(axis=1)
        flagged = df[df["flags"] > 0][["turn", "speaker", "message", "tension", "dominant_emotion"] + flag_cols]

        if flagged.empty:
            st.info("No flagged messages detected. The conversation appears civil.")
        else:
            st.warning(f"Found **{len(flagged)}** flagged message(s).")
            st.dataframe(flagged, use_container_width=True)

        # --- DISPLAY: FULL CHAT LOG ---
        st.markdown("---")
        st.subheader("💬 Full Analyzed Chat")
        display_cols = ["turn", "speaker", "message", "sentiment", "dominant_emotion", "tension", "warmth"]
        st.dataframe(df[display_cols], use_container_width=True)

        # --- DISPLAY: SUMMARY ---
        st.markdown("---")
        st.subheader("📊 Conversation Summary")

        mean_tension = df["tension"].mean()
        early = df.head(max(1, len(df) // 3))["tension"].mean()
        late = df.tail(max(1, len(df) // 3))["tension"].mean()
        change = late - early

        if change > 0.05:
            pattern = "⬆️ Conflict **escalated** over the conversation."
        elif change < -0.05:
            pattern = "⬇️ Conflict **de-escalated** over the conversation."
        else:
            pattern = "➡️ Conflict remained **relatively stable**."

        peak = df.loc[df["tension"].idxmax()]

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Average Tension | **{mean_tension:.3f}** |
        | Peak Tension | **{df['tension'].max():.3f}** at Turn {int(peak['turn'])} by {peak['speaker']} |
        | Early Conflict (first ⅓) | **{early:.3f}** |
        | Late Conflict (last ⅓) | **{late:.3f}** |
        | Overall Pattern | {pattern} |
        """)

        st.info(f"🔍 **Peak conflict message:** \"{peak['message']}\"")
