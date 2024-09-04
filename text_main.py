import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Define paths for saving files in the Downloads folder
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
model_path = os.path.join(downloads_dir, 'emotion_model.keras')
tokenizer_path = os.path.join(downloads_dir, 'tokenizer.pkl')
label_encoder_path = os.path.join(downloads_dir, 'label_encoder.pkl')

# Load and preprocess the dataset
# Adjust delimiter and header settings based on your file format
try:
    data = pd.read_csv("C:/Users/HP/Downloads/textclassification.csv", delimiter=';', header=None)
    data.columns = ["Text", "Emotions"]
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
    raise

# Verify DataFrame structure
print(data.head())
print(data.columns)

if len(data.columns) != 2:
    raise ValueError("The dataset should have two columns: 'Text' and 'Emotions'.")

# Preprocess the text data
data['Text'] = data['Text'].astype(str).str.lower()
data['Text'] = data['Text'].str.replace('[^a-zA-Z\s]', '', regex=True)

# Convert emotions to numeric labels
label_encoder = LabelEncoder()
data['Emotions'] = label_encoder.fit_transform(data['Emotions'])

# Save the label encoder using pickle
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved at {label_encoder_path}")

# Split data into features and labels
X = data['Text']
y = data['Emotions']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad text sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')

# Save the tokenizer using pickle
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved at {tokenizer_path}")

# Build and compile the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# Save the trained model
model.save(model_path)
print(f"Model saved successfully at {model_path}")
