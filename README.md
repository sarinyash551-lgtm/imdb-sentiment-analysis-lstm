# 🎬 IMDb Movie Review Sentiment Analysis

A web-based sentiment analysis application that predicts whether an IMDb movie review is **Positive** or **Negative** using a trained **Long Short-Term Memory (LSTM)** neural network built with TensorFlow and deployed using Streamlit.

---

## 📌 Overview

This project applies Natural Language Processing (NLP) and Deep Learning to classify IMDb movie reviews. Users can enter a review through a simple web interface and receive an instant sentiment prediction.

The model was trained on the IMDb movie review dataset and deployed as an interactive Streamlit application.

---

## ✨ Features

- 🎭 Predicts Positive or Negative movie reviews
- 🧠 LSTM-based Deep Learning model
- ⚡ Fast predictions
- 🌐 Interactive Streamlit web interface
- 📝 Clean and user-friendly UI

---

## 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- IMDb Dataset

---

## 📂 Project Structure

```text
imdb-sentiment-analysis-lstm/
│
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
├── .gitignore
├── .gitattributes
│
├── models/
│   └── imdb_lstm_model.h5
│
├── data/
│   └── imdb_word_index.json
│
└── assets/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/sarinyash551-lgtm/imdb-sentiment-analysis-lstm.git
```

Move into the project

```bash
cd imdb-sentiment-analysis-lstm
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 Usage

1. Launch the Streamlit application.
2. Enter an IMDb movie review.
3. Click **Predict**.
4. View the predicted sentiment.

---

## 🧠 Model

- Model: LSTM Neural Network
- Framework: TensorFlow / Keras
- Task: Binary Sentiment Classification
- Dataset: IMDb Movie Reviews

---

## 📈 Future Improvements

- Confidence score visualization
- Attention-based models
- Transformer-based sentiment analysis
- Docker support
- Cloud deployment

---

## 👨‍💻 Author

**Yash Sarin**

GitHub: https://github.com/sarinyash551-lgtm

---

## 📄 License

This project is licensed under the MIT License.
