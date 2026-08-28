from flask import Flask, render_template, request
import joblib
import re
import string
import sqlite3
from datetime import datetime


app = Flask(__name__)


# ==========================================
# Load ML Model and Vectorizer
# ==========================================

model = joblib.load(
    "model/spam_model.pkl"
)

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# ==========================================
# Database Configuration
# ==========================================

DATABASE = "spam_detector.db"


def get_db_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==========================================
# Initialize Database
# ==========================================

def initialize_database():

    connection = get_db_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            message TEXT NOT NULL,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


# ==========================================
# Text Cleaning
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
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# ==========================================
# Message Analysis
# ==========================================

def analyze_message(text):

    words = text.split()

    url_pattern = r"http\S+|www\S+|https\S+"

    urls = re.findall(
        url_pattern,
        text,
        flags=re.IGNORECASE
    )

    numbers = re.findall(
        r"\d+",
        text
    )

    capital_letters = sum(
        1 for char in text
        if char.isupper()
    )

    exclamation_marks = text.count("!")

    suspicious_keywords = [
        "free",
        "win",
        "winner",
        "won",
        "prize",
        "claim",
        "urgent",
        "offer",
        "cash",
        "reward",
        "congratulations",
        "selected",
        "click",
        "call now",
        "limited",
        "guaranteed"
    ]

    text_lower = text.lower()

    detected_keywords = []

    for keyword in suspicious_keywords:

        if keyword in text_lower:

            detected_keywords.append(
                keyword
            )

    return {

        "characters": len(text),

        "words": len(words),

        "numbers": len(numbers),

        "capital_letters": capital_letters,

        "exclamation_marks": exclamation_marks,

        "contains_url": len(urls) > 0,

        "url_count": len(urls),

        "suspicious_keywords":
            detected_keywords

    }

# ==========================================
# Home Page
# ==========================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None
    confidence = None
    message = ""
    analysis = None


    if request.method == "POST":

        message = request.form.get(
            "message",
            ""
        ).strip()


        if message:

            # ------------------------------
            # Analyze message
            # ------------------------------

            analysis = analyze_message(
                message
            )
        
            # ------------------------------
            # Clean message
            # ------------------------------    
            cleaned_message = clean_text(
                message
            )


            # ------------------------------
            # TF-IDF
            # ------------------------------

            message_vector = vectorizer.transform(
                [cleaned_message]
            )


            # ------------------------------
            # Prediction
            # ------------------------------

            result =    model.predict(
                message_vector
            )[0]


        # ------------------------------
        # Probability Prediction
        # ------------------------------
        
        probabilities = model.predict_proba(
            message_vector
        )[0]
        
        
        # Find the probability of the predicted class
        
        predicted_class_index = list(
            model.classes_
        ).index(result)
        
        
        confidence = round(
            probabilities[predicted_class_index] * 100,
            2
        )
        
        
        # ------------------------------
        # Result
        # ------------------------------
        
        if result == 1:
        
            prediction = "SPAM"
        
        else:
        
            prediction = "NOT SPAM"


            # ------------------------------
            # Save Prediction
            # ------------------------------

            connection = get_db_connection()

            connection.execute(
                """
                INSERT INTO predictions
                (
                    message,
                    prediction,
                    confidence,
                    created_at
                )

                VALUES (?, ?, ?, ?)
                """,

                (
                    message,
                    prediction,
                    confidence,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            connection.commit()

            connection.close()


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message,
        analysis=analysis
    )


# ==========================================
# Prediction History
# ==========================================

@app.route("/history")
def history():

    connection = get_db_connection()

    predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()


    return render_template(
        "history.html",
        predictions=predictions
    )

# ==========================================
# Dashboard
# ==========================================

@app.route("/dashboard")
def dashboard():

    connection = get_db_connection()

    # Total predictions
    total = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        """
    ).fetchone()["count"]


    # Total spam
    spam_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'SPAM'
        """
    ).fetchone()["count"]


    # Total ham
    ham_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'NOT SPAM'
        """
    ).fetchone()["count"]


    # Average confidence
    average_confidence = connection.execute(
        """
        SELECT AVG(confidence) AS average
        FROM predictions
        """
    ).fetchone()["average"]


    # Recent predictions
    recent_predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT 5
        """
    ).fetchall()


    connection.close()


    # Calculate spam rate
    if total > 0:

        spam_rate = round(
            (spam_count / total) * 100,
            2
        )

    else:

        spam_rate = 0


    # Average confidence
    if average_confidence:

        average_confidence = round(
            average_confidence,
            2
        )

    else:

        average_confidence = 0


    return render_template(
        "dashboard.html",

        total=total,

        spam_count=spam_count,

        ham_count=ham_count,

        spam_rate=spam_rate,

        average_confidence=average_confidence,

        recent_predictions=recent_predictions
    )


# ==========================================
# Initialize Database
# ==========================================

initialize_database()


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )