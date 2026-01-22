import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Load saved files (Safe paths)
# ------------------------------
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model(os.path.join(base_dir, "lstm_model.h5"))

    with open(os.path.join(base_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(base_dir, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "❓"

# ------------------------------
# UI Header
# ------------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🧠 Next Word Predictor</h1>
        <p style="font-size:18px;">
            LSTM-based Deep Learning model that predicts the <b>next word</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------
# Input Section (Card Style)
# ------------------------------
with st.container():
    st.markdown("### ✍️ Enter your sentence")
    user_input = st.text_input(
        "",
        placeholder="Example: I am learning machine"
    )

# ------------------------------
# Button + Output
# ------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🚀 Predict Next Word", use_container_width=True)

if predict_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        next_word = predict_next_word(user_input)

        st.markdown(
            f"""
            <div style="
                background-color:#f0f2f6;
                padding:20px;
                border-radius:12px;
                text-align:center;
                margin-top:20px;
            ">
                <h3>✅ Predicted Next Word</h3>
                <h1 style="color:#4CAF50;">{next_word}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------
# Info Section
# ------------------------------
with st.expander("ℹ️ About this App"):
    st.write(
        """
        - Built using **LSTM Neural Network**
        - Implemented with **TensorFlow & Keras**
        - Deployed using **Streamlit Cloud**
        - Predicts the most probable next word based on input text
        """
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("🚀 Made with ❤️ using Deep Learning & Streamlit")
