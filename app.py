from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("model/lstm_model.h5")
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))

with open("model/config.json") as f:
    max_len = json.load(f)["max_len"]

# ---------- PREDICTION ----------
def predict_next_word(text):
    if not text.strip():
        return ""

    seq = tokenizer.texts_to_sequences([text.lower()])[0]
    if len(seq) == 0:
        return ""

    seq = pad_sequences([seq], maxlen=max_len-1, padding="pre")
    pred = np.argmax(model.predict(seq, verbose=0), axis=-1)[0]

    for word, idx in tokenizer.word_index.items():
        if idx == pred:
            return word
    return ""

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/contact")
def contact():
    return "Don't Contact"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    next_word = predict_next_word(text)

    return jsonify({
        "next_word": next_word
    })

if __name__ == "__main__":
    app.run(debug=True,port=5050)
