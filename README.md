# 📰 Fake News Detection using BERT

# 📘 Project Overview

This project detects Fake or Real news using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model.
The system can take any news headline or full article as input and classify it as Fake or Real.
It is designed for both batch predictions via CSV files and interactive single news predictions.
---

# ⚙️ Features

Fine-tuned BERT-base-uncased model for accurate text classification

Works with real-world datasets: True.csv (real news) and Fake.csv (fake news)

Predict news in batch from CSV or one by one interactively

Modular project structure for easy code management: training, prediction, preprocessing

Ready to use in Google Colab or VS Code

Can be extended to a Streamlit/Flask app for live news prediction

---

## 🧩 Folder Structure
```
FakeNewsDetection/
│
├─ fake_news/
│   ├─ scripts/              # Training & prediction scripts
│   │   ├─ train.py          # Train the BERT model
│   │   └─ predict.py        # Run predictions on new news
│   ├─ utils/                # Helper functions & preprocessing
│   │   └─ preprocess.py
│   ├─ fake-news-bert-model/ # Saved trained BERT model
│   ├─ new_news.csv           # Sample CSV with news to predict
│   └─ news_predictions.csv   # Predictions saved after batch run
│
├─ README.md                 # Project overview & instructions
└─ .gitignore                # Ignore unnecessary files (e.g., datasets, checkpoints)

```

---

# 🚀 How to Use

### 🧠 1. Clone the repo:
```bash
git clone https://github.com/kunwardhruv/Fake-news-Detection-.git
cd FakeNewsDetection/fake_news
```

###🔍 2. Install dependencies:
```
python scripts/predict.py

```

### 3.) Train the model (optional if model already trained):
```
python scripts/train.py
```
### 4.) Make predictions:
```
Single news prediction: Edit predict.py or call predict_news("Your news text")
```

### 5.) Batch prediction: Place CSV (new_news.csv) in folder and run:
```
python scripts/predict.py
```
### 6.) Predictions will be saved in news_predictions.csv
---

# 📂 Dataset

True.csv → Real news headlines/articles

Fake.csv → Fake news headlines/articles
---

# 🔧 Requirements

Python >= 3.8

transformers, torch, pandas, numpy

Google Colab recommended for GPU training

---

#⚡ Future Improvements

Deploy as Streamlit or Flask Web App for live news classification

Add multi-lingual news support

Integrate with real-time news APIs
---

# 📊 Example Predictions
```
News Headline	Prediction
Aliens landed in New York today!	Fake
Stock market hits all-time high today.	Real
```
