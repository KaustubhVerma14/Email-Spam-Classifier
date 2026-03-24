# 📧 Email Spam Classifier

## 🚀 Project Overview

This project is an end-to-end Machine Learning system that classifies emails/messages as **Spam** or **Not Spam (Ham)**.
It covers the complete pipeline from **data preprocessing → feature engineering → model training → evaluation → deployment using Gradio**.

---

# 🧠 Approach and Design Choices

## 1️⃣ Data Preprocessing

Raw text data is noisy and inconsistent. To improve model performance, the following preprocessing steps were applied:

* Converted text to lowercase (standardization)
* Removed special characters and numbers (noise removal)
* Tokenized text into words
* Removed stopwords (common words like “the”, “is”)
* Applied stemming using Porter Stemmer (reducing words to root form)

👉 This reduces vocabulary size and helps the model focus on meaningful patterns.

---

## 2️⃣ Feature Engineering (TF-IDF)

Text data was converted into numerical format using **TF-IDF (Term Frequency - Inverse Document Frequency)**.

Why TF-IDF?

* Gives higher importance to meaningful and rare words
* Reduces weight of commonly occurring words
* Improves model’s ability to distinguish spam vs non-spam

---

## 3️⃣ Model Selection

We used **Multinomial Naive Bayes** because:

* Works extremely well for text classification
* Fast and computationally efficient
* Performs well with TF-IDF features

---

## 4️⃣ Modular Code Design

The project was structured into multiple files:

* `preprocessing.py` → Text cleaning
* `vectorizer.py` → TF-IDF conversion
* `train.py` → Model training
* `evaluate.py` → Performance metrics
* `main.py` → Pipeline execution
* `app.py` → Gradio deployment

👉 This modular design improves readability, scalability, and maintainability.

---

# 📊 Model Performance

The model was evaluated using standard classification metrics:

* **Accuracy** → Overall correctness
* **Precision** → Correctness of spam predictions
* **Recall** → Ability to detect actual spam
* **F1 Score** → Balance between precision and recall

### 📈 Sample Results (may vary slightly)

* Accuracy: ~97%
* Precision: ~1.0
* Recall: ~75%
* F1 Score: ~85%

### 🔍 Observations

* The model performs very well on typical spam messages.
* High recall ensures most spam messages are detected.
* Some edge cases (clean-looking spam) may be misclassified.

---

# 🌐 Deployment

The model is deployed using **Gradio**, providing a simple web interface where users can:

* Enter any email/message
* Instantly get prediction:

  * **Spam 🚨**
  * **Not Spam ✅**

---

# ⚙️ Instructions to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd email_spam_classifier
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Training Pipeline

```bash
python main.py
```

This will:

* Train the model
* Save `model.pkl` and `vectorizer.pkl`

---

## 4️⃣ Launch the Web App

```bash
python app.py
```

Open the link shown in terminal (usually):

```
http://127.0.0.1:7860
```

---

# 📁 Project Structure

```
email_spam_classifier/
│
├── preprocessing.py
├── vectorizer.py
├── train.py
├── evaluate.py
├── main.py
├── app.py
├── model.pkl
├── vectorizer.pkl
├── spam.csv
├── requirements.txt
└── README.md
```

---

# 🔮 Future Improvements

* Use advanced models (Logistic Regression, SVM)
* Try deep learning models (LSTM, BERT)
* Add probability/confidence score
* Deploy online (Hugging Face / Render)

---

# 🎯 Conclusion

This project demonstrates how a complete machine learning pipeline can be built from scratch for text classification tasks. It highlights the importance of preprocessing, feature engineering, and proper model evaluation in achieving high performance.

---

# Some screenshots related to the project

![alt text](<Screenshot 2026-03-25 011843.png>)

![alt text](<Screenshot 2026-03-25 011731.png>)

