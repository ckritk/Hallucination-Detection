from flask import Flask, request, jsonify
import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re

# Load the saved model and reducer
model = joblib.load('logistic_model.pkl')
reducer = joblib.load('lda_reducer.pkl')

# Load BERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Text Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(["a", "an", "the", "and", "or", "but"])
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# BERT Embeddings
def get_bert_embeddings(text):
    tokens = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].cpu().numpy()

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get("Prompt", "")
    answer = data.get("Answer", "")
    
    processed_text = preprocess_text(prompt + " " + answer)
    bert_embedding = get_bert_embeddings(processed_text)
    
    # Reduce dimensions
    reduced_data = reducer.transform(bert_embedding)
    
    # Predict
    prediction = model.predict(reduced_data)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=$PORT)
