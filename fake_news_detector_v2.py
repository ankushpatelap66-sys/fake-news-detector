"""
Fake News Detector (Machine Learning / NLP)
----------------------------------------------
Trains a text classification model to predict whether a news
article is "Real" or "Fake", using TF-IDF for text features and
Logistic Regression as the classifier.

Dataset: "Fake and Real News Dataset" (Kaggle)
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Setup:
1. Download the dataset from the link above (Fake.csv and True.csv).
2. Place both files in the same folder as this script.
3. Install requirements:
       pip install pandas scikit-learn
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    """
    Remove dataset-specific artifacts that leak the label instead of
    reflecting real content differences.

    This dataset's "real" articles are almost all Reuters wire stories
    and start with a dateline like "WASHINGTON (Reuters) -". Without
    this cleanup, the model learns to detect the word "Reuters" rather
    than genuinely distinguishing real vs fake news content, so it
    performs poorly on any real-world text that doesn't follow that
    exact format.
    """
    text = re.sub(r"\(Reuters\)", "", text)
    text = re.sub(r"^[A-Z\s]+\s*-\s*", "", text)  # strip leading dateline like "WASHINGTON -"
    return text


def load_dataset():
    """Load and combine the Fake.csv and True.csv files into one labeled dataset."""
    fake_df = pd.read_csv("Fake.csv")
    real_df = pd.read_csv("True.csv")

    fake_df["label"] = "fake"
    real_df["label"] = "real"

    # Combine title + text into a single text field for the model
    fake_df["content"] = fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)
    real_df["content"] = real_df["title"].astype(str) + " " + real_df["text"].astype(str)

    data = pd.concat([fake_df[["content", "label"]], real_df[["content", "label"]]])
    data["content"] = data["content"].apply(clean_text)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

    return data


def train_model(data, sample_size=6000):
    """Convert text to numeric features and train a Logistic Regression classifier."""
    # Using a sample keeps training fast; increase sample_size (or remove it)
    # for a stronger model once you confirm everything works.
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model trained on {len(data)} articles. Test accuracy: {accuracy * 100:.1f}%")
    print("\nDetailed report on test data:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer


def classify_article(model, vectorizer, text):
    """Predict whether a single headline/article is real or fake."""
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()
    return prediction, probability


def main():
    print("===== Fake News Detector =====\n")
    print("Loading dataset...")

    try:
        data = load_dataset()
    except FileNotFoundError:
        print("\nERROR: Fake.csv and/or True.csv not found.")
        print("Download the dataset from Kaggle and place both files in this folder:")
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return

    print(f"Loaded {len(data)} total articles.\n")
    model, vectorizer = train_model(data)

    print("\nType a news headline or article text to check if it's Real or Fake.")
    print("(type 'exit' to quit)")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            print("Goodbye!")
            break

        result, confidence = classify_article(model, vectorizer, text)
        print(f"Prediction: {result.upper()}  (confidence: {confidence * 100:.1f}%)")


if __name__ == "__main__":
    main()
