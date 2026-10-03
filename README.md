# 🧠 CEREBRO // Neural Sentiment Engine
### Tone Intelligence in Conversations

A real-time tone intelligence system that reads the emotional arc of multi-turn conversations — detecting tension, sarcasm, passive-aggression, escalation, and de-escalation patterns.

## Features
- **Multi-turn chat sentiment & tone tracking** — Analyzes every message in a conversation thread
- **Sarcasm, passive-aggression & irony detection** — Uses regex pattern matching + ML emotion classification
- **Emotion arc visualization over conversation time** — Interactive line charts showing tension & warmth trends
- **Speaker-level analysis** — Per-participant conflict and warmth scores
- **Flagged message detection** — Highlights blame, hostility, defensiveness, and withdrawal
- **Works on WhatsApp, Slack, Discord or any chat export**

## Tech Stack
- **Frontend & Backend:** Streamlit (Python)
- **Sentiment Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa fine-tuned on tweets)
- **Emotion Model:** `SamLowe/roberta-base-go_emotions` (28 emotion categories)
- **Pattern Detection:** Custom regex engine for passive-aggression, sarcasm, blame, hostility, defensiveness, withdrawal
- **Scoring:** Weighted composite tension & warmth scores with rolling trend analysis

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Chat Format
Upload a `.txt` file in either format:
```
Name: Message
[Time] Name: Message
```

## Architecture
```
Chat Upload (.txt)
    │
    ├── Regex Parser → DataFrame [turn, speaker, message]
    │
    ├── Sentiment Model (RoBERTa) → positive/negative/neutral
    │
    ├── Emotion Model (GoEmotions) → 28 emotion scores
    │
    ├── Pattern Detector (Regex) → blame, hostility, sarcasm, passive-aggression
    │
    ├── Tension & Warmth Scoring → weighted composite scores
    │
    └── Visualization → Emotion Arc, Speaker Analysis, Flagged Messages
```
