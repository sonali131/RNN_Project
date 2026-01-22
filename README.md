# RNN_Project
Tenserflow+keras+Streamlit
# 🧠 Next Word Prediction using RNN (LSTM)

A deep learning–based **Next Word Prediction** web application built using **TensorFlow, Keras, and Streamlit**. This project uses an **LSTM (Long Short-Term Memory)** model to predict the most probable next word based on a given input sentence.

🌐 **Live Demo:**
👉 [https://rnnproject-srlyrcj9byn9tmdcgstlut.streamlit.app/](https://rnnproject-srlyrcj9byn9tmdcgstlut.streamlit.app/)

---

## ✨ Features

* 🔮 Predicts the **next word** in a sentence
* 🧠 Trained using **Recurrent Neural Network (LSTM)**
* ⚡ Interactive and user-friendly **Streamlit UI**
* 📦 Pre-trained model loaded using `.h5`
* ☁️ Deployed on **Streamlit Cloud**

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow & Keras**
* **NumPy**
* **Streamlit**
* **Pickle** (for tokenizer & metadata)

---

## 📁 Project Structure

```
RNN_Project/
│
├── app.py                 # Streamlit web app
├── lstm_model.h5          # Trained LSTM model
├── tokenizer.pkl          # Saved tokenizer
├── max_len.pkl            # Maximum sequence length
├── qoute_dataset.csv      # Training dataset
├── Sentence.ipynb         # Model training notebook
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sonali131/RNN_Project.git
cd RNN_Project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Model Details

* Model Type: **LSTM (Recurrent Neural Network)**
* Input: Tokenized text sequences
* Output: Probability distribution over vocabulary
* Loss Function: Categorical Crossentropy
* Optimizer: Adam

The model predicts the **most probable next word** using softmax output.

---
UI Screenshot
<img width="840" height="420" alt="new" src="https://github.com/user-attachments/assets/a0ab1419-f083-40b5-96cd-9e238d681c2f" />

## 📊 Dataset

* Custom **quote / sentence dataset** (`qoute_dataset.csv`)
* Preprocessed using tokenization and padding

---

## 📌 Example

**Input:**

```
I am learning machine
```

**Output:**

```
learning
```

---

## 🌱 Future Enhancements

* 🔢 Top-3 word predictions
* 📈 Prediction confidence score
* 🌙 Dark mode UI
* 📱 Mobile-optimized layout
* 🤖 Transformer-based language model

---

## 👩‍💻 Author

**Sonali Mishra**

* GitHub: [https://github.com/sonali131](https://github.com/sonali131)

---

## ⭐ Support

If you like this project, please ⭐ star the repository to show your support!

---

🚀 *Built with passion for Deep Learning & AI*
