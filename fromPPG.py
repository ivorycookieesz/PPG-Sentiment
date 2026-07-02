import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Klasifikasi Sentimen PPG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Klasifikasi Sentimen Kebijakan PPG")
st.write("Masukkan opini untuk memprediksi sentimen.")

MODEL_NAME = "ivorybutter/indobert-ppg-sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

label_map = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

user_input = st.text_area(
    "Masukkan opini tentang PPG",
    height=150,
    placeholder="Contoh: Program PPG sangat membantu meningkatkan kompetensi guru."
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        label = label_map[pred]

        if label == "Positif":
            st.success(f"😊 Sentimen: {label}")
        elif label == "Negatif":
            st.error(f"😠 Sentimen: {label}")
        else:
            st.info(f"😐 Sentimen: {label}")

st.markdown("---")
st.caption("Klasifikasi Sentimen PPG menggunakan IndoBERT")
