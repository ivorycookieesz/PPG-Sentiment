import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Klasifikasi Sentimen PPG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Klasifikasi Sentimen Kebijakan PPG")
st.write("Masukkan opini untuk memprediksi sentimen.")

MODEL_DIR = "ModelPPG"

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Folder '{MODEL_DIR}' tidak ditemukan!")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =====================
# LABEL SENTIMEN
# =====================
# UBAH SESUAI LABEL SAAT TRAINING
label_map = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# =====================
# INPUT
# =====================
user_input = st.text_area(
    "Masukkan opini tentang PPG",
    height=150,
    placeholder="Contoh: Program PPG sangat membantu meningkatkan kompetensi guru."
)

# =====================
# PREDIKSI
# =====================
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
            st.success(f"😊 Sentimen : {label}")

        elif label == "Negatif":
            st.error(f"😠 Sentimen : {label}")

        else:
            st.info(f"😐 Sentimen : {label}")

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption("Klasifikasi Sentimen PPG menggunakan IndoBERT")