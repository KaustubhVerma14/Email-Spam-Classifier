import gradio as gr
import joblib
from preprocessing import preprocess_text

# Load saved model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_spam(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    return "Spam 🚨" if prediction == 1 else "Not Spam ✅"

# Gradio UI
interface = gr.Interface(
    fn=predict_spam,
    inputs="textbox",
    outputs="text",
    title="Email Spam Classifier",
    description="Enter an email or message to check if it's Spam or Not Spam"
)

interface.launch()