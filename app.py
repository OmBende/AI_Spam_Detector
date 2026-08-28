from flask import Flask, render_template, request
import joblib
import re
import string


app = Flask(__name__)


# -----------------------------------------
# Load trained model and vectorizer
# -----------------------------------------

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# -----------------------------------------
# Text cleaning function
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


# -----------------------------------------
# Home page
# -----------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            # Clean message
            cleaned_message = clean_text(message)

            # Convert text into TF-IDF
            message_vector = vectorizer.transform(
                [cleaned_message]
            )

            # Predict
            result = model.predict(message_vector)[0]

            # Get probabilities
            probabilities = model.predict_proba(
                message_vector
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            if result == 1:
                prediction = "SPAM"
            else:
                prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


# -----------------------------------------
# Run Flask application
# -----------------------------------------

if __name__ == "__main__":
    app.run(debug=True)