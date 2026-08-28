# 🛡️ AI Spam Detector

An AI-powered web application that automatically detects whether a text message is **Spam** or **Not Spam** using Machine Learning and Natural Language Processing techniques.

The system uses **TF-IDF** for text feature extraction and compares multiple Machine Learning algorithms before selecting the best-performing model. The final system uses a **Linear Support Vector Machine (Linear SVM)** for spam classification.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain misleading offers, suspicious links, fraudulent claims, or other unwanted content.

The AI Spam Detector provides a simple web interface where users can enter a message and receive:

* Spam / Not Spam classification
* Model confidence score
* Message statistics
* Suspicious keyword detection
* URL detection
* Number detection
* Exclamation mark detection
* Prediction history
* Analytics dashboard

The application also provides user authentication so that each user can maintain their own prediction history and dashboard.

---

## ✨ Features

### 🤖 Machine Learning

* TF-IDF text feature extraction
* Naive Bayes model
* Logistic Regression model
* Linear SVM model
* Automatic model comparison
* Best model selection based on F1 Score
* Saved trained model using Joblib

### 🔍 Message Analysis

The application analyzes messages for:

* URLs
* Numbers
* Capital letters
* Exclamation marks
* Suspicious keywords

Example suspicious keywords include:

* free
* winner
* prize
* claim
* urgent
* offer
* cash
* reward
* congratulations
* selected
* click
* limited
* guaranteed

### 👤 Authentication

* User signup
* User login
* Secure password hashing
* Session-based authentication
* Protected application routes
* Logout functionality
* User-specific prediction history

### 📊 Dashboard

The dashboard displays:

* Total messages analyzed
* Total spam messages
* Total non-spam messages
* Average confidence
* Spam detection rate
* Recent predictions
* Prediction distribution chart
* Machine Learning model information

### 📋 Prediction History

Users can:

* View their previous predictions
* See prediction confidence
* See prediction date and time
* Clear their own prediction history

### 🎯 Confidence

The application displays the model's estimated confidence for each prediction.

> Confidence indicates the model's estimated certainty; it is not a guarantee.

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
Input Message
      ↓
Text Cleaning
      ↓
Lowercase Conversion
      ↓
URL Removal
      ↓
Punctuation Removal
      ↓
Whitespace Normalization
      ↓
TF-IDF Feature Extraction
      ↓
Machine Learning Model
      ↓
Spam / Not Spam Prediction
      ↓
Confidence Score
      ↓
Message Analysis
      ↓
Database Storage
      ↓
Dashboard / History
```

---

## 📚 Dataset

The project uses a labeled SMS spam dataset containing:

**5,572 messages**

The dataset contains two classes:

* `ham` — legitimate message
* `spam` — unwanted/spam message

The dataset is divided using an **80/20 train-test split** with stratification.

```text
Training messages: 4,457
Testing messages: 1,115
```

---

## 🧹 Text Preprocessing

Before training, each message goes through preprocessing.

### 1. Lowercase Conversion

All text is converted to lowercase.

### 2. URL Removal

URLs are removed from the text during model preprocessing.

### 3. Punctuation Removal

Punctuation marks are removed.

### 4. Extra Space Removal

Multiple spaces are converted into a single space.

### 5. TF-IDF

The cleaned text is converted into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

The vectorizer uses a maximum of 5,000 features and English stop-word removal.

---

## 🧪 Models Tested

Three classification algorithms were evaluated:

### 1. Naive Bayes

A probabilistic classification algorithm commonly used for text classification.

### 2. Logistic Regression

A linear classification algorithm suitable for binary classification problems.

### 3. Linear SVM

A Support Vector Machine designed for linear classification.

The final model was selected using the **F1 Score**.

---

## 📊 Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Naive Bayes         |     96.77% |    100.00% |     75.84% |     86.26% |
| Logistic Regression |     96.77% |    100.00% |     75.84% |     86.26% |
| **Linear SVM**      | **97.67%** | **92.41%** | **89.93%** | **91.16%** |

### 🏆 Selected Model

**Linear SVM**

The Linear SVM achieved the highest F1 Score:

**91.16%**

and an accuracy of:

**97.67%**

Therefore, it was selected as the final classification model.

---

## 🔢 Confusion Matrix

The Linear SVM produced the following confusion matrix on the test set:

```text
                  Predicted
                 Ham    Spam

Actual Ham       955     11
Actual Spam       15    134
```

### Interpretation

* **955** legitimate messages were correctly classified.
* **134** spam messages were correctly classified.
* **11** legitimate messages were incorrectly classified as spam.
* **15** spam messages were incorrectly classified as legitimate.

---

## 🛠️ Technology Stack

### Programming Language

* Python 3.x

### Backend

* Flask

### Machine Learning

* Scikit-learn
* TF-IDF
* Linear SVM
* Naive Bayes
* Logistic Regression

### Data Processing

* Pandas
* Regular Expressions

### Database

* SQLite

### Authentication

* Flask Sessions
* Werkzeug Password Hashing

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript
* Chart.js
* Jinja2 Templates

### Model Storage

* Joblib

---

## 📁 Project Structure

```text
AI_Spam_Detector/
│
├── app.py
├── train_model.py
├── README.md
├── .gitignore
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── history.html
│   └── dashboard.html
│
├── static/
│   └── style.css
│
└── spam_detector.db
```

> `spam_detector.db` is generated locally and is excluded from Git using `.gitignore`.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the Project

```bash
cd AI_Spam_Detector
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install flask pandas scikit-learn joblib werkzeug
```

---

## 🧠 Train the Model

Run:

```bash
python train_model.py
```

The training script:

1. Loads the dataset
2. Cleans the messages
3. Converts labels
4. Splits the dataset
5. Applies TF-IDF
6. Trains three ML models
7. Evaluates each model
8. Selects the best model
9. Saves the trained model

Generated files:

```text
model/spam_model.pkl
model/tfidf_vectorizer.pkl
```

---

## ▶️ Run the Application

Start Flask:

```bash
python app.py
```

The application will run locally.

Open the displayed local Flask URL in your browser.

---

## 🔐 Authentication Flow

```text
Signup
  ↓
Account Created
  ↓
Login
  ↓
Session Created
  ↓
Home Page
  ↓
Analyze Messages
  ↓
History / Dashboard
  ↓
Logout
```

Protected routes require the user to be logged in.

---

## 🔍 Explainable Message Analysis

The system provides additional information about the analyzed message.

For example:

```text
Why This Message Was Flagged

🔗 URL Detected
🔢 Numbers Detected
❗ Exclamation Marks
⚠️ Suspicious Keywords

free
winner
claim
prize
```

This analysis helps users understand message characteristics that may be associated with spam.

The analysis does not replace the Machine Learning classifier.

---

## 📊 Dashboard

The dashboard provides an overview of the user's prediction activity.

It includes:

* Total predictions
* Spam count
* Not Spam count
* Spam rate
* Average confidence
* Recent predictions
* Prediction distribution chart
* Model performance information

---

## 🗑️ Clear History

Users can clear their prediction history using:

**Clear My History**

The system deletes only records belonging to the currently logged-in user.

This prevents one user from deleting another user's predictions.

---

## 🔒 Security

The application includes:

* Password hashing using Werkzeug
* Session-based authentication
* Login-protected routes
* Parameterized SQLite queries
* User-specific prediction filtering
* `.env` support for future sensitive configuration

---

## ⚠️ Limitations

The current system has some limitations:

* Performance depends on the training dataset.
* The model is primarily trained on SMS-style messages.
* New spam patterns may not always be detected.
* Confidence is an estimated model certainty, not a guarantee.
* The current application uses a local SQLite database.

---

## 🚀 Future Scope

Possible future improvements include:

* Larger and more diverse datasets
* Deep Learning models
* Transformer-based NLP models
* Email spam detection
* Multilingual spam detection
* URL reputation checking
* Phishing detection
* Real-time email integration
* Cloud deployment
* Advanced explainable AI
* Admin analytics
* API access for external applications

---

## 🎓 Project Objective

The main objective of this project is to develop a practical Machine Learning application capable of automatically classifying messages as spam or legitimate while providing users with useful prediction information and analytics.

---

## 👨‍💻 Project

**AI Spam Detector**

Developed as an academic Machine Learning and Artificial Intelligence project.
