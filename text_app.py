import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os

# Define paths to the saved files
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
model_path = os.path.join(downloads_dir, 'emotion_model.keras')
tokenizer_path = os.path.join(downloads_dir, 'tokenizer.pkl')
label_encoder_path = os.path.join(downloads_dir, 'label_encoder.pkl')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Streamlit app
st.title("Emotion Prediction App")

st.write("Enter some text and get the predicted emotion:")

# Text input from user
input_text = st.text_area("Input Text", "")

if st.button("Predict"):
    if input_text:
        # Preprocess the input text
        input_text = [input_text.lower()]
        input_seq = tokenizer.texts_to_sequences(input_text)
        input_pad = pad_sequences(input_seq, maxlen=100, padding='post')

        # Make prediction
        predictions = model.predict(input_pad)
        predicted_class = predictions.argmax(axis=-1)[0]
        emotion = label_encoder.inverse_transform([predicted_class])[0]

        # Show the result
        st.write(f"Predicted Emotion: {emotion}")
    else:
        st.write("Please enter some text for prediction.")
