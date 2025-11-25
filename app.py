import os
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template

# --------- Configure TF before importing it ---------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # disable oneDNN (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # reduce TF logging

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --------- Load Tokenizer ---------
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# --------- Load Keras .h5 Model instead of TFLite ---------
print("Loading Keras .h5 model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Keras model loaded!")

MAX_SEQUENCE_LENGTH = 200


def preprocess_text(text):
    """Tokenize and pad input text."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded


def scrape_job_description(url):
    """Fetch page HTML and try to extract job description text."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("Error fetching URL:", e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Try to find likely JD containers by id/class
    candidates = []
    keywords = ["job", "description", "jd", "posting", "role", "vacancy", "details"]

    def has_keyword(value):
        if not value:
            return False
        value = value.lower()
        return any(k in value for k in keywords)

    for tag_name in ["div", "section", "article"]:
        candidates.extend(
            soup.find_all(tag_name, id=lambda v: has_keyword(v))
        )
        candidates.extend(
            soup.find_all(tag_name, class_=lambda v: isinstance(v, str) and has_keyword(v))
        )

    if candidates:
        best = max(candidates, key=lambda c: len(c.get_text(" ", strip=True)))
        text = best.get_text(" ", strip=True)
    else:
        # Fallback: just use all paragraphs
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(paras)

    text = text.strip()
    if not text:
        return None

    # Limit length to keep model happy
    return text[:5000]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    combined_text = (request.form.get("combined_text") or "").strip()
    job_url = (request.form.get("job_url") or "").strip()

    # If no text but URL provided -> scrape from URL
    if not combined_text and job_url:
        scraped = scrape_job_description(job_url)
        if not scraped:
            return render_template(
                "index.html",
                prediction="Could not extract job description from that URL. Try pasting the text manually."
            )
        combined_text = scraped

    # If still nothing, ask user
    if not combined_text:
        return render_template(
            "index.html",
            prediction="Please enter a job description or paste a job URL."
        )

    # Preprocess the input
    input_data = preprocess_text(combined_text)

    # Keras model prediction
    prob = float(model.predict(input_data)[0][0])

    # Threshold
    result = "Fraudulent" if prob > 0.7 else "Legitimate"

    return render_template(
        "index.html",
        prediction=f"The job post is {result} (score: {prob:.3f})"
    )


if __name__ == "__main__":
    app.run(debug=True)
