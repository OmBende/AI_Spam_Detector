from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import re
import string
import sqlite3
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = "ai-spam-detector-secret-key"

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
# Login Required
# ==========================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper

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

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL UNIQUE,

            email TEXT NOT NULL UNIQUE,

            password TEXT NOT NULL

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
# Signup
# ==========================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        if not username or not email or not password:

            flash(
                "All fields are required."
            )

            return redirect(
                url_for("signup")
            )


        connection = get_db_connection()


        existing_user = connection.execute(
            """
            SELECT id
            FROM users
            WHERE username = ? OR email = ?
            """,

            (
                username,
                email
            )
        ).fetchone()


        if existing_user:

            connection.close()

            flash(
                "Username or email already exists."
            )

            return redirect(
                url_for("signup")
            )


        hashed_password = generate_password_hash(
            password
        )


        connection.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password
            )

            VALUES (?, ?, ?)
            """,

            (
                username,
                email,
                hashed_password
            )
        )


        connection.commit()

        connection.close()


        flash(
            "Account created successfully. Please login."
        )

        return redirect(
            url_for("login")
        )


    return render_template(
        "signup.html"
    )

# ==========================================
# Login
# ==========================================

# ==========================================
# Login
# ==========================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        connection.close()

        # ----------------------------------
        # Check Login Credentials
        # ----------------------------------

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]

            session["username"] = user["username"]

            flash(
                "Login successful!",
                "success"
            )

            return redirect(
                url_for("home")
            )

        else:

            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

    return render_template(
        "login.html"
    )
# ==========================================
# Logout
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "Logout successful!",
        "logout"
    )

    return redirect(
        url_for("login")
    )

# ==========================================
# Home Page
# ==========================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
@login_required
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

            result = model.predict(
                message_vector
            )[0]


            # ------------------------------
            # Probability Prediction
            # ------------------------------

            probabilities = model.predict_proba(
                message_vector
            )[0]


            # Find probability of predicted class

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
                    created_at,
                    user_id
                )
        
                VALUES (?, ?, ?, ?, ?)
                """,
        
                (
                    message,
                    prediction,
                    confidence,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    session["user_id"]
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
@login_required
def history():

    connection = get_db_connection()

    predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        """,

        (session["user_id"],)
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
@login_required
def dashboard():

    connection = get_db_connection()

    # Total predictions
    total = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()["count"]


    # Total spam
    spam_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'SPAM' AND user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()["count"]


    # Total ham
    ham_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE prediction = 'NOT SPAM'
        AND user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()["count"]


    # Average confidence
    average_confidence = connection.execute(
        """
        SELECT AVG(confidence) AS average
        FROM predictions
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()["average"]


    # Recent predictions
    recent_predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 5
        """,
        (session["user_id"],)
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
