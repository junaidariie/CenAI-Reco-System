import torch
import torch.nn as nn
import numpy as np
import re

# =========================== Device ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model running on: {DEVICE}")

# =========================== Tokenizer ===========================
def simple_tokenize(text):
    return text.split()

# =========================== Model ===========================
class GoEmotionsLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, num_classes=28, num_layers=2):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x)
        _, (h, _) = self.lstm(x)

        h_forward = h[-2]
        h_backward = h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)

        h_cat = self.dropout(h_cat)
        out = self.fc(h_cat)
        return out

# =========================== Load Model (ONCE) ===========================
def load_goemotion_model(path="artifacts/goemotions_bilstm_checkpoint.pth"):
    checkpoint = torch.load(path, map_location=DEVICE)

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = GoEmotionsLSTM(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, vocab, max_len

MODEL, VOCAB, MAX_LEN = load_goemotion_model()

# =========================== Emotion Map ===========================
EMOTION_MAP = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

# =========================== Sentiment Groups ===========================
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief"
}

NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
}

NEUTRAL_EMOTIONS = {
    "confusion", "curiosity", "realization", "surprise", "neutral"
}

# =========================== Preprocessing ===========================
_CLEAN_RE = re.compile(r'[^a-z0-9\s]+')

def clean_text(text: str) -> str:
    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())

# =========================== Core Prediction ===========================
@torch.inference_mode()
def predict_sentiment(text: str):
    text = clean_text(text)
    tokens = simple_tokenize(text)

    seq = [VOCAB.get(tok, 1) for tok in tokens]  # 1 = <UNK>

    if len(seq) < MAX_LEN:
        seq.extend([VOCAB["<PAD>"]] * (MAX_LEN - len(seq)))
    else:
        seq = seq[:MAX_LEN]

    x = torch.tensor(seq, dtype=torch.long, device=DEVICE).unsqueeze(0)

    logits = MODEL(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    pos_score = 0.0
    neg_score = 0.0
    neu_score = 0.0

    for i, p in enumerate(probs):
        emotion = EMOTION_MAP[i]
        if emotion in POSITIVE_EMOTIONS:
            pos_score += p
        elif emotion in NEGATIVE_EMOTIONS:
            neg_score += p
        elif emotion in NEUTRAL_EMOTIONS:
            neu_score += p

    scores = {
        "positive": pos_score,
        "negative": neg_score,
        "neutral": neu_score
    }

    sentiment = max(scores, key=scores.get)
    confidence = float(scores[sentiment] / (pos_score + neg_score + neu_score + 1e-8))

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }

# =========================== Public Function ===========================
def find_sentiment(text: str):
    return predict_sentiment(text)

# =========================== Analyze Sentiment ===========================

def analyze_reviews_sentiment(reviews: list[str]):
    """
    reviews: list of review strings
    returns: percentage distribution
    """

    total = len(reviews)

    if total == 0:
        return {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0
        }

    counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }

    for review in reviews:
        result = find_sentiment(review)
        counts[result["sentiment"]] += 1

    percentages = {
        "positive": round((counts["positive"] / total) * 100, 2),
        "negative": round((counts["negative"] / total) * 100, 2),
        "neutral": round((counts["neutral"] / total) * 100, 2)
    }

    return percentages


"""TEST_REVIEWS_50 = [
    # Positive (1–18)
    "Absolutely loved this movie, the story and acting were brilliant.",
    "One of the best films I have seen this year, totally worth it.",
    "The cinematography was stunning and the soundtrack was perfect.",
    "I really enjoyed every minute of it, great experience.",
    "An amazing performance by the lead actor, truly outstanding.",
    "This movie exceeded my expectations in every way.",
    "Beautiful storytelling and emotional depth, loved it.",
    "The direction and screenplay were top-notch.",
    "A शानदार movie, very entertaining and engaging.",
    "I was smiling the whole time, such a feel-good film.",
    "The action sequences were incredible and well choreographed.",
    "A masterpiece, will definitely watch it again.",
    "The chemistry between the actors was amazing.",
    "Really inspiring and motivational movie.",
    "This film made my day, absolutely fantastic.",
    "Loved the humor and the emotional moments.",
    "A very satisfying and enjoyable watch.",
    "Brilliant execution and great visuals.",

    # Negative (19–36)
    "This movie was a complete waste of time.",
    "I did not like it at all, very boring and slow.",
    "The plot made no sense and the acting was bad.",
    "Terrible screenplay and weak performances.",
    "I was very disappointed with this film.",
    "The movie felt too long and dragged a lot.",
    "Poor direction and horrible editing.",
    "Not worth the hype, very average experience.",
    "The story was predictable and dull.",
    "I regret watching this movie.",
    "Bad acting and cringe dialogues.",
    "This film was really annoying to watch.",
    "Nothing interesting happened in the entire movie.",
    "The worst movie I have seen in a long time.",
    "Very weak script and poor execution.",
    "It was painful to sit through this movie.",
    "Extremely disappointing and underwhelming.",
    "The movie failed to impress in any aspect.",

    # Neutral (37–50)
    "The movie was okay, nothing special.",
    "It was an average film with decent acting.",
    "The story was simple and straightforward.",
    "Some parts were good, some parts were boring.",
    "It was a one-time watch kind of movie.",
    "The film was neither good nor bad.",
    "Decent movie, could have been better.",
    "The acting was fine and the story was okay.",
    "Nothing extraordinary, just a regular film.",
    "It was watchable but not memorable.",
    "An average experience overall.",
    "The movie did its job, nothing more.",
    "It was fine for a weekend watch.",
    "Neither impressive nor terrible."
]

def test_50_reviews_sentiment():
    print("=" * 80)
    print("TESTING SENTIMENT DISTRIBUTION ON 50 MOVIE REVIEWS")
    print("=" * 80)

    # Individual predictions (optional but good for debugging)
    for idx, review in enumerate(TEST_REVIEWS_50, start=1):
        result = find_sentiment(review)
        print(f"{idx:02d}. {review}")
        print(f"    → Sentiment: {result['sentiment'].upper():8} | Confidence: {result['confidence']}")
        print("-" * 80)

    print("\nAGGREGATED RESULT")
    print("=" * 80)

    distribution = analyze_reviews_sentiment(TEST_REVIEWS_50)

    print(f"Positive : {distribution['positive']}%")
    print(f"Negative : {distribution['negative']}%")
    print(f"Neutral  : {distribution['neutral']}%")
    print("=" * 80)


# Run test
if __name__ == "__main__":
    test_50_reviews_sentiment()
"""