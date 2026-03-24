import pandas as pd
from pathlib import Path

from preprocessing import preprocess_text
from vectorizer import get_tfidf_features
from train import train_model
from evaluate import evaluate_model

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "spam.csv"

df = pd.read_csv(file_path, encoding='latin-1')

# Clean dataset
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess
df['processed'] = df['message'].apply(preprocess_text)

# TF-IDF
X, tfidf = get_tfidf_features(df['processed'])
y = df['label']

# Train
model, X_test, y_test = train_model(X, y)

# Evaluate
accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

"""import joblib

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")"""