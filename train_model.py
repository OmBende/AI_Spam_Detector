import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

data = pd.read_csv(
    "dataset/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))


# -----------------------------------------
# 2. Text Cleaning Function
# -----------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


data["clean_message"] = data["message"].apply(clean_text)


# -----------------------------------------
# 3. Convert Labels
# -----------------------------------------

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------------
# 4. Split Dataset
# -----------------------------------------

X = data["clean_message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# -----------------------------------------
# 5. TF-IDF Vectorization
# -----------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------------
# 6. Train Naive Bayes Model
# -----------------------------------------

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("Model trained successfully!")


# -----------------------------------------
# 7. Evaluate Model
# -----------------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------
# 8. Save Model
# -----------------------------------------

joblib.dump(
    model,
    "model/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")