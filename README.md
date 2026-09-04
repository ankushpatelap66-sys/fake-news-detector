# Fake News Detector

A machine learning model that classifies news articles as **Real** or **Fake** using Natural Language Processing.

## Features
- Trains a Logistic Regression classifier on a real-world dataset of 44,000+ news articles
- Uses TF-IDF vectorization to convert article text into numeric features
- Identifies and corrects a data leakage issue in the dataset (the model was initially relying on the presence of the word "Reuters" as a shortcut instead of learning genuine content patterns)
- Reports accuracy, precision, and recall on a held-out test set
- Interactive CLI to test custom headlines/articles

## Tech Stack
- Python
- scikit-learn
- pandas

## Dataset
[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — download `Fake.csv` and `True.csv` and place them in this folder.

## How to Run
```bash
pip install pandas scikit-learn
python fake_news_detector_v2.py
```

## What I Learned
- How to identify and fix **data leakage**, where a model learns a spurious pattern (a dataset artifact) instead of the actual signal
- The difference between training on short text (headlines) vs. long text (full articles), and how that affects model confidence
- Practical experience with TF-IDF and Logistic Regression for text classification

## Example
```
Enter text: WASHINGTON (Reuters) - Floods hit Nepal after heavy monsoon rainfall
Prediction: REAL (confidence: 91.2%)
```
