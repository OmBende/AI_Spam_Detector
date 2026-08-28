import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv(
    "dataset/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("=" * 60)
print("AI SPAM DETECTOR - MODEL TRAINING")
print("=" * 60)

print(f"\nTotal messages: {len(data)}")


# ==========================================
# 2. Text Cleaning
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


data["clean_message"] = data["message"].apply(
    clean_text
)


# ==========================================
# 3. Convert Labels
# ==========================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 4. Train/Test Split
# ==========================================

X = data["clean_message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")


# ==========================================
# 5. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF vectorization completed.")


# ==========================================
# 6. Define Models
# ==========================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Linear SVM": LinearSVC()

}


# ==========================================
# 7. Train and Evaluate
# ==========================================

results = {}

best_model = None
best_model_name = None
best_f1 = 0


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Ham",
                "Spam"
            ]
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # Select best model based on F1 score
    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name


# ==========================================
# 8. Display Final Comparison
# ==========================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 70)

for name, metrics in results.items():

    print(
        f"{name:<25}"
        f"{metrics['accuracy'] * 100:<15.2f}"
        f"{metrics['precision'] * 100:<15.2f}"
        f"{metrics['recall'] * 100:<15.2f}"
        f"{metrics['f1'] * 100:<15.2f}"
    )


# ==========================================
# 9. Save Best Model
# ==========================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Selected Model: {best_model_name}"
)

print(
    f"F1 Score: {best_f1 * 100:.2f}%"
)


joblib.dump(
    best_model,
    "model/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\nBest model saved successfully!")

print("Model path:")
print("model/spam_model.pkl")

print("\nVectorizer saved successfully!")

print("Vectorizer path:")
print("model/tfidf_vectorizer.pkl")