
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import os

model_path = "../models/fake-news-bert-model"
assert os.path.isdir(model_path), "Model folder does not exist!"
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model.eval()

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = np.argmax(outputs.logits.detach().numpy(), axis=1)[0]
    return "Real" if pred == 1 else "Fake"

csv_path = "../data/new_news.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['prediction'] = df['text'].apply(predict_news)
    output_path = "../outputs/news_predictions.csv"
    df.to_csv(output_path, index=False)
    print("Predictions saved at:", output_path)
