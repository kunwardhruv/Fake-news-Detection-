
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datasets import Dataset

# Load dataset
real = pd.read_csv("../data/True.csv")
fake = pd.read_csv("../data/Fake.csv")
real['label'] = 1
fake['label'] = 0
df = pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)
df = df[['text','label']]

# HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.train_test_split(test_size=0.2)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
tokenized_datasets = hf_dataset.map(tokenize, batched=True)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=500,
    save_strategy="epoch",
    report_to=[]
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train & Save
trainer.train()
trainer.save_model("../models/fake-news-bert-model")
tokenizer.save_pretrained("../models/fake-news-bert-model")
