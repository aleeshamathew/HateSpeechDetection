from flask import Flask, render_template, request
import re
import string
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# 1️⃣ Initialize Flask
# -------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------
# 2️⃣ Load the trained model and tokenizer
# -------------------------------------------------------
MODEL_PATH = "best_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"   # or tokenizer.joblib if you used joblib

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100   # Adjust this to match your model training

# -------------------------------------------------------
# 3️⃣ Text Cleaning Function
# -------------------------------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                     # Mentions
    text = re.sub(r"#\w+", "", text)                     # Hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)                      # Numbers
    text = re.sub(r"\s+", " ", text).strip()             # Extra spaces
    return text

# -------------------------------------------------------
# 4️⃣ Sentiment Prediction Function
# -------------------------------------------------------
def predict_sentiment(text):
    cleaned_text = clean_tweet(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]

    # assuming model output = [neg, neu, pos]
    label_idx = np.argmax(pred)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[label_idx]

# -------------------------------------------------------
# 5️⃣ Flask Routes
# -------------------------------------------------------
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    sentiment = predict_sentiment(user_input)
    return render_template('output.html', text=user_input, sentiment=sentiment)

# -------------------------------------------------------
# 6️⃣ Run the App
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
