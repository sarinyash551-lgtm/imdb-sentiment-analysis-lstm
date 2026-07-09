# app.py
# IMDB Movie Review Classifier by YashSarin
# Simple Streamlit app using the LSTM model trained in Colab

import json
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load the model and word index
# ----------------------------

model = keras.models.load_model("imdb_lstm_Yashsarin.h5", compile=False)


with open("imdb_word_index.json", "r") as f:
    word_index = json.load(f)

# settings
vocab_size = 10000
max_len = 200
index_from = 3      # 0=PAD, 1=START, 2=UNK, 3=UNUSED


# ----------------------------
# Helper function
# ----------------------------

def review_to_sequence(text, word_index, max_len=200):
    """
    Convert raw text review to padded number sequence.
    """
    words = text.lower().split()
    seq = []

    for w in words:
        if w in word_index and word_index[w] + index_from < vocab_size:
            seq.append(word_index[w] + index_from)
        else:
            seq.append(2)   # unknown

    padded = pad_sequences([seq], maxlen=max_len, padding='pre', truncating='pre')
    return padded


def predict_review(text):
    """
    Run model prediction and return label + probability.
    """
    x = review_to_sequence(text, word_index, max_len)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob


# ----------------------------
# Streamlit UI
# ----------------------------

def main():
    st.title("IMDB Movie Review Classifier by YashSarin")

    st.write("Enter a movie review below and click Predict.")

    user_input = st.text_area("Movie review:", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please write something.")
        else:
            label, prob = predict_review(user_input)
            st.subheader("Prediction")
            st.write("Label:", label)
            st.write("Probability:", prob)

    st.write("---")
    st.subheader("Example Reviews (5)")

    example_reviews = [
        "This movie was amazing and I really enjoyed it.",
        "I did not like this movie. It was boring.",
        "The story was fine but acting was weak.",
        "Terrible film. I will not watch again.",
        "Nice movie with good music and good performance."
    ]

    for i, rev in enumerate(example_reviews, start=1):
        label, prob = predict_review(rev)
        st.write(f"Review {i}: {rev}")
        st.write(f" => {label} (prob = {prob:.4f})")
        st.write("")


if __name__ == "__main__":
    main()


