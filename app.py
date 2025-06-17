import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    try:
        model.load_state_dict(torch.load("bert_stress_model.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    except FileNotFoundError:
        st.error("Error: 'bert_stress_model.pt' not found. Please ensure the model file is in the correct location.")
        st.stop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

def predict_stress(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities.cpu().numpy()[0]

st.title("Stress Level Prediction")

user_input = st.text_area("Enter text to analyze for stress:")

if st.button("Predict"):
    if user_input:
        probabilities = predict_stress(user_input)
        stress_probability = probabilities[1]
        no_stress_probability = probabilities[0]

        st.write(f"Probability of Stress: {stress_probability:.4f}")
        st.write(f"Probability of No Stress: {no_stress_probability:.4f}")

        if stress_probability > no_stress_probability:
            st.title("The model predicts the text indicates stress.")
        else:
            st.title("The model predicts the text indicates no stress.")

        # Corrected progress bar usage:
        st.progress(float(stress_probability)) # or stress_probability.item()
    else:
        st.warning("Please enter some text to analyze.")